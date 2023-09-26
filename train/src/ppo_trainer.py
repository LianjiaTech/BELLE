import time
from typing import Callable, List, Union, Optional
from pathlib import Path
import torch
from trl import PPOTrainer
from trl.core import (
    logprobs_from_logits,
    WANDB_PADDING,
    PPODecorators,
    convert_to_scalar,
    stack_dicts,
    stats_to_np,
)
from src.models.generation_utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from src.utils import bind_methods_from_class_to_instance, get_ds_state_dict
import torch.distributed as dist
import numpy as np


class PPOTrainerForZero3(PPOTrainer):
    def _generate_batched(
        self,
        query_tensors: List[torch.Tensor],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: int = None,
        remove_padding: bool = True,
        **generation_kwargs,
    ):
        outputs = []

        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)

        for i in range(0, len(query_tensors), batch_size):
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.current_device)

            unwrap_model = self.accelerator.unwrap_model(self.model).pretrained_model
            bind_methods_from_class_to_instance(
                unwrap_model,
                GenerationMixin,
                include=[
                    "contrastive_search",
                    "greedy_search",
                    "sample",
                    "beam_search",
                    "beam_sample",
                    "group_beam_search",
                    "constrained_beam_search",
                    "assisted_decoding",
                    "generate",
                ],
            )
            generations = unwrap_model.generate(
                deepspeed_model=self.model,
                **padded_inputs,
                **generation_kwargs,
                synced_gpus=True,
            )
            # generations = self.accelerator.unwrap_model(self.model).generate(
            #     **padded_inputs, **generation_kwargs
            # )

            for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                if not self.is_encoder_decoder:
                    output = generation[(1 - mask).sum() :]  # remove padding
                else:
                    output = generation

                if not return_prompt and not self.is_encoder_decoder:
                    output = output[(mask).sum() :]  # remove prompt

                if remove_padding and self.tokenizer.eos_token_id in output:
                    pad_mask = output == self.tokenizer.eos_token_id
                    pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                    output = output[: pad_start + 1]  # keep the eos token at the end

                outputs.append(output)

        self.tokenizer.padding_side = padding_side_default
        return outputs

    @PPODecorators.empty_cuda_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        response_masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            response_masks (List[`torch.FloatTensor`], *optional*)):
                List of tensors containing masks of the response tokens.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.config.batch_size

        queries, responses, scores, response_masks = self._step_safety_checker(
            bs, queries, responses, scores, response_masks
        )
        scores = torch.tensor(scores, device=self.current_device)
        if self.config.use_score_scaling:
            # Score scaling
            scores_mean, scores_std = self.running.update(scores)
            tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
            score_scaling_factor = (
                self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
            )
            if self.config.use_score_norm:
                scores = (
                    scores - self.running.mean.to(**tensor_to_kwargs)
                ) / score_scaling_factor
            else:
                scores /= score_scaling_factor

        if self.config.score_clip is not None:
            # Score clipping
            scores_dtype = scores.dtype
            scores = torch.clip(
                scores.float(), -self.config.score_clip, self.config.score_clip
            ).to(dtype=scores_dtype)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = scores.mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs[
                    "decoder_input_ids"
                ] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs[
                    "decoder_attention_mask"
                ] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        with torch.no_grad():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                model_inputs,
                response_masks=response_masks,
                return_logits=full_kl_penalty,
            )
            # for when the model is a peft model
            if self.is_peft_model and hasattr(
                self.accelerator.unwrap_model(self.model).pretrained_model,
                "disable_adapter",
            ):
                with self.accelerator.unwrap_model(
                    self.model
                ).pretrained_model.disable_adapter():
                    ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                        self.model,
                        queries,
                        responses,
                        model_inputs,
                        return_logits=full_kl_penalty,
                    )
            elif self.is_peft_model and not hasattr(
                self.model.pretrained_model, "disable_adapter"
            ):
                raise ValueError(
                    "You are using a `peft` version that does not support `disable_adapter`. Please update your `peft` version to the latest version."
                )

            else:
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.ref_model,
                    queries,
                    responses,
                    model_inputs,
                    return_logits=full_kl_penalty,
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(
                    logits_or_none, None, gather=False
                )
                ref_full_logprobs = logprobs_from_logits(
                    ref_logits_or_none, None, gather=False
                )

                rewards, non_score_reward = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks
                )
            else:
                rewards, non_score_reward = self.compute_rewards(
                    scores, all_logprobs, ref_logprobs, masks
                )
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(
                values, rewards, masks
            )
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = (
                    backward_batch_start + self.config.backward_batch_size
                )
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(
                    0, self.config.backward_batch_size, self.config.mini_batch_size
                ):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[
                        mini_batch_start:mini_batch_end
                    ]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [
                            batch_dict["responses"][i] for i in mini_batch_inds
                        ],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {
                            k: mini_batch_dict[k] for k in model_inputs_names
                        }

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if self.is_deepspeed_zero3_enabled():
                    this_peer_finished_flag = torch.tensor(
                        0.0 if not early_stop else 1.0
                    ).to(self.current_device)
                    dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                    if this_peer_finished_flag.item() > 0.0:
                        break
                elif early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(
            train_stats["policy/advantages"]
        ).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(
            train_stats["policy/advantages"], WANDB_PADDING
        )
        train_stats["policy/ratio"] = torch.flatten(
            train_stats["policy/ratio"]
        ).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        config: Optional[dict] = None,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        is_deepspeed_used = (
            self.accelerator.distributed_type == "DEEPSPEED"
            and hasattr(self.accelerator.state, "deepspeed_plugin")
        )
        if not is_deepspeed_used:
            if self.accelerator.is_main_process:
                super().save_pretrained(
                    save_directory,
                    config=config,
                    repo_id=repo_id,
                    push_to_hub=push_to_hub,
                    **kwargs,
                )
        else:
            if self.is_deepspeed_zero3_enabled():
                state_dict = get_ds_state_dict(self.model)
            else:
                # Only run on rank 0 except stage 3
                if self.accelerator.is_main_process:
                    state_dict = get_ds_state_dict(self.model)

            if self.accelerator.is_main_process:
                unwrap_model: PreTrainedModel = self.accelerator.unwrap_model(
                    self.model
                ).pretrained_model
                unwrap_model.save_pretrained(save_directory, state_dict=state_dict)

    def is_deepspeed_zero3_enabled(self):
        return (
            self.accelerator.state.deepspeed_plugin.deepspeed_config[
                "zero_optimization"
            ]["stage"]
            == 3
        )
