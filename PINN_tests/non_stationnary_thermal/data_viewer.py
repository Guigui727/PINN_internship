import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class data_viewer():
    def __init__(self, data, max_val, units):
        self.extent=[0., max_val[0], 0., max_val[1]]
        self.param = np.linspace(0., max_val[2], data.shape[2])
        self.units = units
        self.data = data


    

    def show_fig(self):

        fig, ax = plt.subplots()
        img = plt.imshow(self.data[:, :, self.param.shape[0] - 1].T, extent=self.extent, vmax=self.data.max(), vmin=self.data.min())
        ax.set_xlabel(self.units[0])
        ax.set_ylabel(self.units[1])

        cax = plt.axes([0.12, 0.17, 0.78, 0.03])
        plt.colorbar(cax=cax, orientation='horizontal')

        axparam = plt.axes([0.12, 0.08, 0.78, 0.03])
        param_slider = Slider(
            ax=axparam,
            label=self.units[2],
            valmin=0.,
            valmax=self.param[-1],
            valinit=self.param[-1],
            valstep=self.param
        )

        def update(param):
            indice_param = int(param * (self.param.shape[0] - 1) / self.param[-1])
            img.set_data(self.data[:, :, indice_param].T)
            fig.canvas.draw_idle()

        param_slider.on_changed(update)

        plt.show()

    
if __name__ == "__main__":
    data = np.load('pregenerated_data.npy')
    data = data.reshape((100, 25, 100))

    dv = data_viewer(data, [1.1, 0.4, 20.], ["X (m)", "Y(m)", "T(s)"])
    dv.show_fig()


