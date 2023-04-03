from collections import deque
from typing import Deque

from common.Graph import MxGraph
from core import ax, mx

from .MxDataGenerator import MxDataGenerator
from .MxModel import MxModel


class MxModelTrainer(mx.Disposable):

    def __init__(self, src_gen : MxDataGenerator, dst_gen : MxDataGenerator, model : MxModel, state : dict = None):
        super().__init__()
        state = state or {}

        self._tg = ax.TaskGroup().dispose_with(self)
        self._main_thread = ax.get_current_thread()
        self._training_tg = ax.TaskGroup().dispose_with(self)
        self._training_thread = ax.Thread(name='MxModelTrainer:training_thread').dispose_with(self)

        self._src_gen = src_gen
        self._dst_gen = dst_gen
        self._model = model

        self._mx_metrics_graph = MxGraph(state=state.get('metrics_graph_state', None)).dispose_with(self)#

        self._mx_error = mx.TextEmitter().dispose_with(self)
        self._mx_batch_size = mx.Number(state.get('batch_size', 4), config=mx.NumberConfig(min=1, max=64, step=1)).dispose_with(self)
        self._mx_batch_acc = mx.Number(state.get('batch_acc', 1), config=mx.NumberConfig(min=1, max=512, step=1)).dispose_with(self)
        self._mx_learning_rate = mx.Number(state.get('learning_rate', 50), config=mx.NumberConfig(min=1, max=1000, step=1)).dispose_with(self)

        self._mx_mse_power = mx.Number(state.get('mse_power', 1.0), config=mx.NumberConfig(min=0.0, max=1.0, step=0.1, decimals=1)).dispose_with(self)
        self._mx_dssim_x4_power = mx.Number(state.get('dssim_x4_power', 1.0), config=mx.NumberConfig(min=0.0, max=1.0, step=0.1, decimals=1)).dispose_with(self)
        self._mx_dssim_x8_power = mx.Number(state.get('dssim_x8_power', 1.0), config=mx.NumberConfig(min=0.0, max=1.0, step=0.1, decimals=1)).dispose_with(self)
        self._mx_dssim_x16_power = mx.Number(state.get('dssim_x16_power', 1.0), config=mx.NumberConfig(min=0.0, max=1.0, step=0.1, decimals=1)).dispose_with(self)
        self._mx_dssim_x32_power = mx.Number(state.get('dssim_x32_power', 1.0), config=mx.NumberConfig(min=0.0, max=1.0, step=0.1, decimals=1)).dispose_with(self)

        self._mx_training = mx.Flag(False).dispose_with(self)
        self._mx_iteration_time = mx.Property[float](0.0).dispose_with(self)

        self.set_training(state.get('training', False))

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
    def mx_dssim_x4_power(self) -> mx.INumber: return self._mx_dssim_x4_power
    @property
    def mx_dssim_x8_power(self) -> mx.INumber: return self._mx_dssim_x8_power
    @property
    def mx_dssim_x16_power(self) -> mx.INumber: return self._mx_dssim_x16_power
    @property
    def mx_dssim_x32_power(self) -> mx.INumber: return self._mx_dssim_x32_power

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
                'dssim_x4_power' : self._mx_dssim_x4_power.get(),
                'dssim_x8_power' : self._mx_dssim_x8_power.get(),
                'dssim_x16_power' : self._mx_dssim_x16_power.get(),
                'dssim_x32_power' : self._mx_dssim_x32_power.get(),
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

        src_gen = self._src_gen
        dst_gen = self._dst_gen
        model = self._model

        src_dg_tasks = ax.TaskSet[MxDataGenerator.GenResult]()
        src_dg_data : Deque[MxDataGenerator.GenResult] = deque()

        dst_dg_tasks = ax.TaskSet[MxDataGenerator.GenResult]()
        dst_dg_data : Deque[MxDataGenerator.GenResult] = deque()

        training_tasks = ax.TaskSet[MxModel.StepResult]()

        iteration_time = 0
        while True:
            yield ax.switch_to(self._main_thread)

            batch_size = self._mx_batch_size.get()
            batch_acc = self._mx_batch_acc.get()
            lr = self._mx_learning_rate.get() * 1e-6

            mse_power = self._mx_mse_power.get()
            dssim_x4_power = self._mx_dssim_x4_power.get()
            dssim_x8_power = self._mx_dssim_x8_power.get()
            dssim_x16_power = self._mx_dssim_x16_power.get()
            dssim_x32_power = self._mx_dssim_x32_power.get()

            self._mx_iteration_time.set(iteration_time)

            yield ax.switch_to(self._training_thread)

            # Keep data_gen tasks to fill the buffer
            for _ in range(src_gen.workers_count*2 - src_dg_tasks.count - len(src_dg_data)):
                src_dg_tasks.add( src_gen.generate(  batch_size=batch_size,
                                                        W =(res := model.get_input_resolution()),
                                                        H = res,
                                                        grayscale=model.get_input_ch()==1) ) #

            for _ in range(dst_gen.workers_count*2 - dst_dg_tasks.count - len(dst_dg_data)):
                dst_dg_tasks.add( dst_gen.generate(  batch_size=batch_size,
                                                        W =(res := model.get_input_resolution()),
                                                        H = res,
                                                        grayscale=model.get_input_ch()==1) ) #

            # Collect generated data
            for task in src_dg_tasks.fetch(succeeded=True):
                src_dg_data.append(task.result)

            for task in dst_dg_tasks.fetch(succeeded=True):
                dst_dg_data.append(task.result)

            # Keep running two training tasks
            if training_tasks.count < 2 and len(src_dg_data) != 0 and len(dst_dg_data) != 0:
                src_data = src_dg_data.popleft()
                dst_data = dst_dg_data.popleft()

                training_tasks.add( model.step(MxModel.StepRequest(
                                                src_image_np        = src_data.image_deformed_np,
                                                src_target_image_np = src_data.image_np,
                                                src_target_mask_np  = src_data.target_mask_np,

                                                dst_image_np        = dst_data.image_deformed_np,
                                                dst_target_image_np = dst_data.image_np,
                                                dst_target_mask_np  = dst_data.target_mask_np,

                                                dssim_x4_power=dssim_x4_power,
                                                dssim_x8_power=dssim_x8_power,
                                                dssim_x16_power=dssim_x16_power,
                                                dssim_x32_power=dssim_x32_power,
                                                mse_power=mse_power,
                                                masked_training=True,

                                                batch_acc=batch_acc,
                                                lr=lr, )))

            for t in training_tasks.fetch(succeeded=True):
                t_result = t.result
                iteration_time = t_result.time
                self._mx_metrics_graph.add({ '@(Metric.Error) src' : t_result.error_src,
                                             '@(Metric.Error) dst' : t_result.error_dst,
                                             '@(Metric.Accuracy) src' : t_result.accuracy_src,
                                             '@(Metric.Accuracy) dst' : t_result.accuracy_dst,
                                             '@(Metric.Iteration_time)' : iteration_time,
                                            } )



            # Cancel training due to failed tasks.
            for task in src_dg_tasks.fetch(succeeded=False) | dst_dg_tasks.fetch(succeeded=False) | training_tasks.fetch(succeeded=False):
                yield ax.switch_to(self._main_thread)

                self._mx_training.set(False)
                self._mx_error.emit(str(task.error))
                return

            yield ax.sleep(0)
