from peft import PeftModel
from transformers.trainer import *

from src.utils import get_ds_state_dict


class MyTrainer(Trainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Add supports for peft + deepspeed zero 3

        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
            ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
            or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
            or self.fsdp is not None
            or self.is_fsdp_enabled
        ):
            if self.is_fsdp_enabled:
                os.makedirs(output_dir, exist_ok=True)
                self.accelerator.state.fsdp_plugin.save_model(self.accelerator, self.model, output_dir)
            else:
                state_dict = self.model.state_dict()

                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
        elif self.is_deepspeed_enabled:
            # This must be called on all ranks in stage 3
            if is_deepspeed_zero3_enabled():
                state_dict = get_ds_state_dict(self.deepspeed)
            else:
                # Only run on rank 0 except stage 3
                if self.args.should_save:
                    state_dict = get_ds_state_dict(self.deepspeed)
            # this takes care of everything as long as we aren't under zero3
            # Only run on rank 0
            if self.args.should_save:           
                # state_dict is available on rank 0     
                self._save(output_dir, state_dict=state_dict)
            
        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """
        Add supports for peft resume
        """
        if model is None:
            model = self.model

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)

        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        adapter_model_path = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        safe_adapter_model_path = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)

        if not any(
            os.path.isfile(f) for f in [weights_file, safe_weights_file, adapter_model_path, weights_index_file, safe_weights_index_file, safe_adapter_model_path]
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file) or \
            os.path.isfile(adapter_model_path) or os.path.isfile(safe_adapter_model_path):
            # If the model is on the GPU, it still works!
            if is_sagemaker_mp_enabled():
                if os.path.isfile(os.path.join(resume_from_checkpoint, "user_content.pt")):
                    # If the 'user_content.pt' file exists, load with the new smp api.
                    # Checkpoint must have been saved with the new smp api.
                    smp.resume_from_checkpoint(
                        path=resume_from_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False
                    )
                else:
                    # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                    # Checkpoint must have been saved with the old smp api.
                    if hasattr(self.args, "fp16") and self.args.fp16 is True:
                        logger.warning(
                            "Enabling FP16 and loading from smp < 1.10 checkpoint together is not suppported."
                        )
                    state_dict = torch.load(weights_file, map_location="cpu")
                    # Required for smp to not auto-translate state_dict from hf to smp (is already smp).
                    state_dict["_smp_is_partial"] = False
                    load_result = model.load_state_dict(state_dict, strict=True)
                    # release memory
                    del state_dict
            elif self.is_fsdp_enabled:
                self.accelerator.state.fsdp_plugin.load_model(self.accelerator, model, resume_from_checkpoint)
            else:
                if is_peft_available() and isinstance(model, PeftModel):
                    model.load_adapter(resume_from_checkpoint, getattr(model, "active_adapter", "default"), is_trainable=True)
                else:
                    # We load the model state dict on the CPU to avoid an OOM error.
                    if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                        state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
                    else:
                        state_dict = torch.load(weights_file, map_location="cpu")

                    # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                    # which takes *args instead of **kwargs
                    load_result = model.load_state_dict(state_dict, False)
                    # release memory
                    del state_dict
                    self._issue_warnings_after_load(load_result)
        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(
                model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled(), prefer_safe=self.args.save_safetensors
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)
