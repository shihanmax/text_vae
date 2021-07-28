import logging

import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
from nlkit.trainer import BaseTrainer
from nlkit.utils import Phase, check_should_do_early_stopping
from nltk.tokenize import word_tokenize
from .utils import translate_text, translate_idx
from .config import Config


logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    
    def __init__(
        self, model, train_data_loader, valid_data_loader, test_data_loader, 
        lr_scheduler, optimizer, weight_init, summary_path, device, criterion,
        total_epoch, model_path, gradient_clip, verbose, 
        not_early_stopping_at_first, es_with_no_improvement_after, 
        sampling_text_list, vocab,
    ):
        super(Trainer, self).__init__(
            model, train_data_loader, valid_data_loader, test_data_loader, 
            lr_scheduler, optimizer, weight_init, summary_path, device, 
            criterion, total_epoch, model_path,
        )
        self.ce_loss = CrossEntropyLoss()
        self.valid_loss_record = []
        self.gradient_clip = gradient_clip
        self.verbose = verbose
        self.not_early_stopping_at_first = not_early_stopping_at_first
        self.es_with_no_improvement_after = es_with_no_improvement_after
        self.sampling_text_list = sampling_text_list
        self.vocab = vocab
        
    def forward_model(self, x, valid_length, phase: Phase):
        # outputs, indices, mean, log_var
        on_train = phase is Phase.TRAIN
        return self.model(x, valid_length, on_train)
    
    def do_sample(self):
        print("==Do sampling now==")
        for text in self.sampling_text_list:
            target_indices, valid_length = translate_text(
                text, self.vocab, Config.max_src_length, word_tokenize,
            )
            target_indices = torch.tensor(target_indices).unsqueeze(0).to(
                Config.device
            )
            valid_length = torch.tensor(valid_length).unsqueeze(0)
            
            _, gen_indices, _, _ = self.forward_model(
                target_indices, valid_length, Phase.VALID,
            )
            
            output_text = translate_idx(gen_indices, self.vocab)
            
            print("raw text: {}\n gen text: {}".format(
                text, output_text
            ))

    def do_random_sample(self, n_samples=3):
        print("==Do random sampling now==")
        for _ in range(n_samples):
            z = torch.randn(1, Config.z_dim).to(Config.device)
            _, indices = self.model.decoder(z)
            output_text = translate_idx(indices, self.vocab)
            
            print("\ngen text: {}".format(output_text))
            
    def calc_loss(self, mean, log_var, target, output):
        re_construct_loss = self.ce_loss(output.permute(0, 2, 1), target)
        
        kl_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return re_construct_loss, kl_loss
    
    def iteration(self, epoch, data_loader, phase):
        data_iter = tqdm(
            enumerate(data_loader),
            desc="EP:{}:{}".format(phase.name, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}",
        )

        total_loss = []
        valid_loss = []
        
        for idx, data in data_iter:
            if phase is Phase.TRAIN:
                self.global_train_step += 1
            elif phase is Phase.VALID:
                self.global_valid_step += 1
            else:
                self.global_test_step += 1

            # data to device
            data = {key: value.to(self.device) for key, value in data.items()}

            # forward the model
            if phase == Phase.TRAIN:
                outputs, indices, mean, log_var = self.forward_model(
                    data["x"], data["valid_length"], phase,
                )
                
            else:
                with torch.no_grad():
                    outputs, indices, mean, log_var = self.forward_model(
                        data["x"], data["valid_length"], phase,
                    )
     
            re_construct_loss, kl_loss = self.calc_loss(
                mean, log_var, data["x"], outputs,
            )
            
            loss = kl_loss + re_construct_loss
            
            total_loss.append(loss.item())
            valid_loss.append(re_construct_loss.item())
            
            # do backward if on train
            if phase == Phase.TRAIN:
                self.optimizer.zero_grad()
                loss.backward()

                if self.gradient_clip:
                    clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip,
                    )
                    
                self.optimizer.step()

            log_info = {
                "phase": phase.name,
                "epoch": epoch,
                "iter": idx,
                "curr_loss": loss.item(),
                "curr_re_con_loss": re_construct_loss.item(),
                "curr_kl_loss": kl_loss.item(),
                "avg_loss": sum(total_loss) / len(total_loss),
            }

            if self.verbose and not idx % self.verbose:
                data_iter.write(str(log_info))
      
        if phase is Phase.TRAIN:
            self.lr_scheduler.step()  # step every train epoch
            self.do_sample()
            self.do_random_sample()
            
        avg_loss = sum(total_loss) / len(total_loss)
        avg_valid_loss = sum(valid_loss) / len(valid_loss)
        
        logger.info(
            "EP:{}_{}, avg_loss={}".format(
                epoch,
                phase.name,
                avg_loss,
            ),
        )

        # 记录训练信息
        record = {
            "epoch": epoch,
            "status": phase.name,
            "avg_loss": avg_loss,
            "avg_valid_loss": avg_valid_loss,
        }

        self.train_record.append(record)

        # check should early stopping at valid
        if phase is Phase.VALID:
            self.metric_record_on_valid.append(avg_valid_loss)

            should_stop = check_should_do_early_stopping(
                self.metric_record_on_valid,
                self.not_early_stopping_at_first,
                self.es_with_no_improvement_after,
                acc_like=False,
            )

            if should_stop:
                best_epoch = should_stop
                logger.info("Now stop training..")
                return best_epoch
        
        return False
