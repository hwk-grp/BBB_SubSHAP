import torch
import os

OUTPUT_DIR = "preds"

def save_best_checkpoint(model, optimizer, best_epoch, best_score, j):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_epoch': best_epoch,
        'best_score': best_score
    }
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_path = os.path.join(ckpt_dir, f'best_ckpt' + str(j) + '.pth')
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at epoch {best_epoch} with score {best_score:.4f}")


def load_checkpoint(model, optimizer, filepath="checkpoint.pth"):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Checkpoint loaded from epoch {epoch} with loss {loss:.4f}")
    return model, optimizer, epoch, loss

def load_best_result(model, j):
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    best_ckpt_path = os.path.join(ckpt_dir, f'best_ckpt' + str(j) + '.pth')
    ckpt = torch.load(best_ckpt_path)
    model.load_state_dict(ckpt['model'])
    best_epoch = ckpt['best_epoch']

    return model
