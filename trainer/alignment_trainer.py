import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CTCLoss
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

from models.aligner import Aligner
from models.forward_tacotron import ForwardTacotron
from trainer.common import Averager, TTSSession, MaskedL1
from utils import hparams as hp
from utils.checkpoints import save_checkpoint
from utils.dataset import get_tts_datasets
from utils.decorators import ignore_exception
from utils.display import stream, simple_table, plot_mel
from utils.dsp import reconstruct_waveform, rescale_mel, np_now
from utils.paths import Paths
from utils.text import sequence_to_text


class AlignmentTrainer:

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.writer = SummaryWriter(log_dir=paths.forward_log, comment='v1')
        self.ctc_loss = CTCLoss()

    def train(self, model: Aligner, optimizer: Optimizer) -> None:
        for i, session_params in enumerate(hp.aligner_schedule, 1):
            lr, max_step, bs = session_params
            if model.get_step() < max_step:
                train_set, val_set = get_tts_datasets(
                    path=self.paths.data, batch_size=bs, r=1, model_type='aligner')
                session = TTSSession(
                    index=i, r=1, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, optimizer, session)

    def train_session(self, model: ForwardTacotron,
                      optimizer: Optimizer, session: TTSSession) -> None:
        current_step = model.get_step()
        training_steps = session.max_step - current_step
        total_iters = len(session.train_set)
        epochs = training_steps // total_iters + 1
        simple_table([(f'Steps', str(training_steps // 1000) + 'k Steps'),
                      ('Batch Size', session.bs),
                      ('Learning Rate', session.lr)])

        for g in optimizer.param_groups:
            g['lr'] = session.lr

        loss_avg = Averager()
        duration_avg = Averager()
        device = next(model.parameters()).device  # use same device as model parameters
        for e in range(1, epochs + 1):
            for i, (x, m, ids, mel_lens, seq_lens) in enumerate(session.train_set, 1):

                start = time.time()
                model.train()
                x, m, mel_lens, seq_lens = x.to(device), m.to(device), \
                                                mel_lens.to(device), seq_lens.to(device)

                m = m.transpose(1, 2)
                model.train()
                pred = model(m)
                pred = pred.transpose(0, 1).log_softmax(2)
                loss = self.ctc_loss(pred, x, mel_lens, seq_lens)
                loss_avg.add(loss)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step = model.get_step()
                k = step // 1000

                duration_avg.add(time.time() - start)
                speed = 1. / duration_avg.get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {loss_avg.get():#.4} ' \
                      f'| {speed:#.2} steps/s | Step: {k}k | '

                first_pred = pred.transpose(0, 1)[0].max(1)[1].detach().cpu().numpy().tolist()
                first_pred_d = sequence_to_text(first_pred)
                first_target = x[0].detach().cpu().numpy().tolist()
                first_target_d = sequence_to_text(first_target)

                if model.get_step() % 1000 == 0:
                    print()
                    print(f'pred dec: {first_pred_d}')
                    print(f'target dec: {first_target_d}')

                if step % hp.forward_checkpoint_every == 0:
                    ckpt_name = f'forward_step{k}K'
                    save_checkpoint('forward', self.paths, model, optimizer,
                                    name=ckpt_name, is_silent=True)

                self.writer.add_scalar('Loss/train', loss, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                stream(msg)

            save_checkpoint('aligner', self.paths, model, optimizer, is_silent=True)
            loss_avg.reset()
            duration_avg.reset()
            print(' ')
