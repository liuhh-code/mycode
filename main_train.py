import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel


def parse_agrs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_biomedbert', action='store_true')
    parser.add_argument('--lr_bert_fine_tune', type=float, default=5e-6)

    parser.add_argument('--image_dir', type=str, default='/home/sun/data/iu_xray/images/')
    parser.add_argument('--ann_path', type=str, default='/home/sun/data/iu_xray/annotation.json')

    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'])
    parser.add_argument('--max_seq_length', type=int, default=60)
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--visual_extractor', type=str, default='resnet101')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True)

    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--d_vf', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--logit_layers', type=int, default=1)
    parser.add_argument('--bos_idx', type=int, default=0)
    parser.add_argument('--eos_idx', type=int, default=0)
    parser.add_argument('--pad_idx', type=int, default=0)
    parser.add_argument('--use_bn', type=int, default=0)
    parser.add_argument('--drop_prob_lm', type=float, default=0.5)
    
    parser.add_argument('--rm_num_slots', type=int, default=3)
    parser.add_argument('--rm_num_heads', type=int, default=8)
    parser.add_argument('--rm_d_model', type=int, default=512)

    parser.add_argument('--sample_method', type=str, default='beam_search')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sample_n', type=int, default=1)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--output_logsoftmax', type=int, default=1)
    parser.add_argument('--decoding_constraint', type=int, default=0)
    parser.add_argument('--block_trigrams', type=int, default=1)

    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='results/iu_xray')
    parser.add_argument('--record_dir', type=str, default='records/')
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'])
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4')
    parser.add_argument('--early_stop', type=int, default=50)

    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr_ve', type=float, default=5e-5)
    parser.add_argument('--lr_ed', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--amsgrad', type=bool, default=True)

    parser.add_argument('--lr_scheduler', type=str, default='StepLR',
                        choices=['StepLR', 'warmup_cosine', 'warmup_linear'])
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--warmup_steps', type=int, default=1000)

    parser.add_argument('--seed', type=int, default=9233)
    parser.add_argument('--resume', type=str)

    parser.add_argument('--ngram_size', type=int, default=0)
    parser.add_argument('--repetition_penalty_weight', type=float, default=0.0)

    args = parser.parse_args()
    return args


def main():
    args = parse_agrs()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    tokenizer = Tokenizer(args)

    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    model = R2GenModel(args, tokenizer)

    criterion = compute_loss
    metrics = compute_scores

    optimizer = build_optimizer(args, model)
    
    total_steps = args.epochs * len(train_dataloader)
    
    lr_scheduler = build_lr_scheduler(args, optimizer, total_steps)

    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                      test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()