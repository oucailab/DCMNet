from parameter import *
from net.all import DCMNet

model = DCMNet()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("using cuda:0 as device")
else:
    print("using cpu as device")


if __name__ == '__main__':
    total_loss = 0
    max_acc = 0
    for epoch in range(500):
        model.train_start()
        n = 0
        model.train(epoch)
        model.val_start()
        acc1 = model.cal_acc(epoch)
        if acc1 > max_acc:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }, 'model/' + train_dataset + '_' + str(acc1) + '.pth')
            max_acc = acc1

    acc1 = model.cal_acc(0)
