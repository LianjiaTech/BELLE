from peft import PeftModel
from transformers.trainer import *

from src.utils import get_ds_state_dict


class MyTrainer(Trainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Add supports for peft

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
        ):
            state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
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

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Add supports for peft
        """
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, (PreTrainedModel, PeftModel)):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), (PreTrainedModel, PeftModel)):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    