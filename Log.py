import logging.config
import os
import time

class Log:
    def __init__(self, config_path, log_dir, res_dir, adjust_dir=None, global_level="DEBUG"):
        """
        :param config_path:
        :param log_dir
        """
        log_dir, res_dir = [_ + time.strftime("-%Y-%m-%d-%H-%M-%S") if os.path.exists(_) else _ for _ in (log_dir, res_dir)]
        self.log_dir = log_dir
        self.res_dir = res_dir
        self.adjust_dir = adjust_dir
        self.global_level = global_level.upper()
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        logging.config.fileConfig(
            config_path,
            defaults={'log_dir': self.log_dir, 'res_dir':self.res_dir, 'adjust_dir': self.adjust_dir}
        )
        
        
        self.res_logger = logging.getLogger('res_logger')
        self.weight_logger = logging.getLogger('weight_logger')
        self.args_logger = logging.getLogger('args_logger')
        self.loss_logger = logging.getLogger('loss_logger')
        self.log_logger = logging.getLogger('log_logger')
        self.adjust_logger = logging.getLogger('adjust_logger')

        
        self._set_global_level()

    def _set_global_level(self):
        if self.global_level != 'DEBUG':
            self.weight_logger.setLevel(logging.CRITICAL)
            # self.args_logger.setLevel(logging.CRITICAL)
            self.loss_logger.setLevel(logging.CRITICAL)
            self.log_logger.setLevel(logging.CRITICAL)

    def log_result(self, message: str):
        self.res_logger.info(message)

    def log_weight(self, message: str):
        self.weight_logger.debug(message)

    def log_args(self, message: str):
        self.args_logger.debug(message)

    def log_loss(self, message: str):
        self.loss_logger.debug(message)

    def log_info(self, message: str):
        self.log_logger.debug(message)

    def log_adjust(self, message: str):
        self.adjust_logger.info(message)