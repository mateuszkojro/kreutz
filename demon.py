"""
Zaawansowane Metody Obliczeniowe - zadanie 2

- Implementacja demona Creutz'a

Mateusz Kojro
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from tabulate import tabulate


def field_sum(fields):
    return fields[2] * (fields[0] + fields[1] + fields[3] + fields[4]) / 2


class Creutz:
    """
    Implementacja Demona Creutza
    """

    def __init__(self, shape=(10, 10), demon_energy=10):
        # self.array = np.random.choice([-1, 1], size=shape)
        # self.array = np.ones(shape=shape)
        self.shape = shape
        self.array = np.full(shape, -1)
        self.demon_energy = demon_energy

    @staticmethod
    def energy_sum(array):
        filtered_image = ndimage.generic_filter(array,
                                                field_sum,
                                                mode="reflect",
                                                footprint=[
                                                    [0, 1, 0],
                                                    [1, 1, 1],
                                                    [0, 1, 0],
                                                ]).ravel()
        # plt.ion()
        # plt.imshow(filtered_image.reshape(array.shape))
        # redraw()
        
        return -np.sum(filtered_image)

    def magnetyzacja(self):
        return np.abs(np.sum(self.array.ravel()))

    def sum(self):
        return self.energy_sum(self.array)

    def step(self):
        temp = self.array.copy()
        energy_before = self.energy_sum(temp)

        # FIXME: That could be done better
        rand_idx = random.randint(0, self.shape[0] * self.shape[1] - 1)
        temp.ravel()[rand_idx] *= -1

        energy_after = self.energy_sum(temp)
        energy_diff = energy_before - energy_after

        print(
            tabulate(
                [[self.demon_energy, energy_before, energy_after, energy_diff]],
                headers=[
                    "Demon energy", "energy_before", "energy_after",
                    "energy_diff"
                ]))

        if self.demon_energy + energy_diff >= 0:
            self.demon_energy += energy_diff
            self.array = temp.copy()
        return self.demon_energy


def redraw():
    plt.gcf().canvas.flush_events()
    plt.show(block=False)


def main():
    max_e = 100
    demon = Creutz(shape=(100, 100), demon_energy=max_e)

    plt.imshow(demon.array)
    plt.colorbar()
    plt.show()
    max_t = 200

    magnetzyzacje = []
    demon_e = []
    time_range = list(range(max_t))

    for _ in time_range:
        magnetzyzacje.append(demon.magnetyzacja())
        demon_e.append(demon.demon_energy)
        demon.step()

    plt.plot(time_range, demon_e, label="Energia demona")
    plt.legend()

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    bx = fig.add_subplot(2, 2, 2)
    cx = fig.add_subplot(2, 2, 3)
    dx = fig.add_subplot(2, 2, 4)

    ax.plot(time_range, demon_e, label="Energia demona")
    ax.legend()

    bx.hist(demon_e, label="Energia demona - histogram", bins=20)
    bx.legend()

    cx.plot(time_range, magnetzyzacje, label="Magnetyzacja")
    cx.legend()

    dx.hist(magnetzyzacje, label="Magnetyzacje - histogram", bins=20)
    dx.legend()

    plt.show()


if __name__ == "__main__":
    main()
