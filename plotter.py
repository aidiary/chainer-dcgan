import matplotlib.pyplot as plt
import json


def plot_result(logfile, targets=['gen/loss', 'dis/loss'], outfile=None):
    fig, ax = plt.subplots()

    result = json.load(open(logfile))
    for target in targets:
        epoch = []
        loss = []
        for x in result:
            epoch.append(x['iteration'])
            loss.append(x[target])
        ax.plot(epoch, loss, label=target, marker='.')

    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.legend(loc='best')
    ax.grid(True)
    fig.tight_layout()

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()


def main():
    plot_result('result/log', ['gen/loss', 'dis/loss'], 'dcgan_loss.png')


if __name__ == '__main__':
    main()
