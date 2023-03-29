from collections import deque
from typing import Deque

from common.Graph import MxGraph
from core import ax, mx

from .MxDataGenerator import MxDataGenerator
from .MxModel import MxModel


class MxModelTrainer(mx.Disposable):

    def __init__(self, data_generator : MxDataGenerator, model : MxModel, state : dict = None):
        super().__init__()
        state = state or {}

        self._tg = ax.TaskGroup().dispose_with(self)
        self._main_thread = ax.get_current_thread()
        self._training_tg = ax.TaskGroup().dispose_with(self)
        self._training_thread = ax.Thread(name='MxModelTrainer:training_thread').dispose_with(self)

        self._data_generator = data_generator
        self._model = model

        self._mx_metrics_graph = MxGraph(state=state.get('metrics_graph_state', None)).dispose_with(self)#

        self._mx_error = mx.TextEmitter().dispose_with(self)
        self._mx_batch_size = mx.Number(state.get('batch_size', 4), config=mx.NumberConfig(min=1, max=64, step=1)).dispose_with(self)
        self._mx_batch_acc = mx.Number(state.get('batch_acc', 1), config=mx.NumberConfig(min=1, max=512, step=1)).dispose_with(self)
        self._mx_learning_rate = mx.Number(state.get('learning_rate', 250), config=mx.NumberConfig(min=1, max=1000, step=1)).dispose_with(self)
        self._mx_train_encoder = mx.Flag(state.get('train_encoder', True)).dispose_with(self)
        self._mx_train_decoder = mx.Flag(state.get('train_decoder', True)).dispose_with(self)
        
        self._mx_mse_power = mx.Number(state.get('mse_power', 1.0), config=mx.NumberConfig(min=0.0, max=1.0, step=0.1, decimals=1)).dispose_with(self)
        self._mx_dssim_power = mx.Number(state.get('dssim_power', 0.0), config=mx.NumberConfig(min=0.0, max=1.0, step=0.1, decimals=1)).dispose_with(self)
        self._mx_training = mx.Flag(False).dispose_with(self)
        self._mx_iteration_time = mx.Property[float](0.0).dispose_with(self)

        self.set_training(state.get('training', False))


    @property
    def mx_data_generator(self) -> MxDataGenerator: return self._data_generator
    @property
    def mx_model(self) -> MxModel: return self._model
    @property
    def mx_metrics_graph(self) -> MxGraph:
        return self._mx_metrics_graph
    @property
    def mx_error(self) -> mx.ITextEmitter_r: return self._mx_error
    @property
    def mx_batch_size(self) -> mx.INumber: return self._mx_batch_size
    @property
    def mx_batch_acc(self) -> mx.INumber: return self._mx_batch_acc
    @property
    def mx_learning_rate(self) -> mx.INumber: return self._mx_learning_rate
    @property
    def mx_mse_power(self) -> mx.INumber: return self._mx_mse_power
    @property
    def mx_dssim_power(self) -> mx.INumber: return self._mx_dssim_power
    @property
    def mx_train_encoder(self) -> mx.IFlag: return self._mx_train_encoder
    @property
    def mx_train_decoder(self) -> mx.IFlag: return self._mx_train_decoder
    
    @property
    def mx_training(self) -> mx.IFlag_r:
        """Indicates training or not"""
        return self._mx_training
    @property
    def mx_iteration_time(self) -> mx.IProperty_r[float]:
        """indicates time of last iteration"""
        return self._mx_iteration_time

    @ax.task
    def get_state(self) -> dict:
        yield ax.attach_to(self._tg)

        metrics_graph_t = self._mx_metrics_graph.get_state()
        yield ax.wait(metrics_graph_t)
        if not metrics_graph_t.succeeded:
            yield ax.cancel(metrics_graph_t.error)

        return {'batch_size' : self._mx_batch_size.get(),
                'batch_acc' : self._mx_batch_acc.get(),
                'learning_rate' : self._mx_learning_rate.get(),
                'mse_power' : self._mx_mse_power.get(),
                'dssim_power' : self._mx_dssim_power.get(),
                'train_encoder' : self._mx_train_encoder.get(),
                'train_decoder' : self._mx_train_decoder.get(),
                'training' : self._mx_training.get(),
                'metrics_graph_state' : metrics_graph_t.result,
                }


    @ax.protected_task
    def set_training(self, b : bool):
        """"""
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._training_thread)
        yield ax.attach_to(self._training_tg, cancel_all=True)
        yield ax.switch_to(self._main_thread)

        if b:
            self._mx_training.set(True)
        else:
            self._mx_training.set(False)
            return

        data_generator = self._data_generator
        model = self._model

        dg_tasks = ax.TaskSet[MxDataGenerator.GenResult]()
        dg_data : Deque[MxDataGenerator.GenResult] = deque()
        training_tasks = ax.TaskSet[MxModel.TrainStepResult]()

        iteration_time = 0
        while True:
            yield ax.switch_to(self._main_thread)

            batch_size = self._mx_batch_size.get()
            batch_acc = self._mx_batch_acc.get()
            lr = self._mx_learning_rate.get() * 1e-6
            train_encoder = self._mx_train_encoder.get()
            train_decoder = self._mx_train_decoder.get()

            mse_power = self._mx_mse_power.get()
            dssim_power = self._mx_dssim_power.get()

            self._mx_iteration_time.set(iteration_time)

            yield ax.switch_to(self._training_thread)

            # Keep data_gen tasks to fill the buffer
            for _ in range(data_generator.workers_count*2 - dg_tasks.count - len(dg_data)):
                dg_tasks.add( data_generator.generate(  batch_size=batch_size,
                                                        W =(res := model.get_input_resolution()),
                                                        H = res,
                                                        grayscale=model.get_input_ch()==1) ) #
            # Collect generated data
            for task in dg_tasks.fetch(succeeded=True):
                dg_data.append(task.result)

            # Keep running two training tasks
            if training_tasks.count < 2 and len(dg_data) != 0:
                data = dg_data.popleft()
                training_tasks.add( model.train_step(   image_np=data.image_np,
                                                        target_mask_np=data.target_mask_np,
                                                        mse_power=mse_power,
                                                        dssim_x4_power=dssim_power,
                                                        dssim_x8_power=dssim_power,
                                                        dssim_x16_power=dssim_power,
                                                        dssim_x32_power=dssim_power,
                                                        batch_acc=batch_acc,
                                                        lr=lr,
                                                        train_encoder=train_encoder,
                                                        train_decoder=train_decoder, 
                                                        ))

            for t in training_tasks.fetch(succeeded=True):
                t_result = t.result
                iteration_time = t_result.step_time
                self._mx_metrics_graph.add({ '@(Metric.Error)' : t_result.error,
                                             '@(Metric.Accuracy)' : t_result.accuracy,
                                             '@(Metric.Iteration_time)' : iteration_time,
                                            } )



            # Cancel training due to failed tasks.
            for task in dg_tasks.fetch(succeeded=False) | training_tasks.fetch(succeeded=False):
                yield ax.switch_to(self._main_thread)

                self._mx_training.set(False)
                self._mx_error.emit(str(task.error))
                return

            yield ax.sleep(0)
