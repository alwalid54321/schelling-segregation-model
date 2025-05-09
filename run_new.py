# run.py
import argparse
from model import SchellingModel
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=20)
    parser.add_argument('--height', type=int, default=20)
    parser.add_argument('--density', type=float, default=0.8)
    parser.add_argument('--homophily', type=float, default=0.3)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    model = SchellingModel(
        width=args.width,
        height=args.height,
        density=args.density,
        homophily=args.homophily
    )
    unhappy_history = []
    for _ in range(args.steps):
        model.step()
        unhappy_history.append(len(model.unhappy_agents) / len(model.agents))

    if args.plot:
        plt.figure()
        plt.plot(unhappy_history)
        plt.xlabel('Step')
        plt.ylabel('Unhappy Fraction')
        plt.title('Unhappy Agents Over Time')
        plt.show()


if __name__ == '__main__':
    main()
