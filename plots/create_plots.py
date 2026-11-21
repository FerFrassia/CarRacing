import matplotlib.pyplot as plt
import pandas as pd

def create_plots():
    trainDF = pd.read_csv('./train.csv')
    trainDF['return_avg'] = trainDF['return'].rolling(100, min_periods=1).mean()
    
    evalDF = pd.read_csv('./eval.csv')

    plt.plot(trainDF['step'], trainDF['return_avg'], c='b', lw=2, label='Train (average every 100 episodes)')
    plt.plot(evalDF['step'], evalDF['eval_return'], c='g', lw=1, marker='s', ms=3, label='Eval')
    plt.xlabel('step')
    plt.ylabel('return')
    plt.title('Rewards')
    plt.xscale('log')
    plt.xlim(1e5, None)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig('Rewards.png')


def main():
    create_plots()

if __name__ == '__main__':
    main()