import logging.config
import os
import time

class Log:
    def __init__(self, config_path, log_dir, res_dir, adjust_dir=None, global_level="DEBUG"):
        """
        初始化日志类
        :param config_path: 配置文件路径（如 'logging.conf'）
        :param log_dir: 日志文件存储目录
        """
        log_dir, res_dir = [_ + time.strftime("-%Y-%m-%d-%H-%M-%S") if os.path.exists(_) else _ for _ in (log_dir, res_dir)]
        # if os.path.exists(log_dir): log_dir = log_dir + time.strftime("-%Y-%m-%d-%H-%M-%S")
        # if os.path.exists(log_dir): log_dir = log_dir + time.strftime("-%Y-%m-%d-%H-%M-%S")
        self.log_dir = log_dir
        self.res_dir = res_dir
        self.adjust_dir = adjust_dir
        self.global_level = global_level.upper()
        
        # 创建日志目录（如果不存在）
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 动态加载配置文件，注入 log_dir 变量
        logging.config.fileConfig(
            config_path,
            defaults={'log_dir': self.log_dir, 'res_dir':self.res_dir, 'adjust_dir': self.adjust_dir}
        )
        
        # 初始化各个日志记录器
        self.res_logger = logging.getLogger('res_logger')
        self.weight_logger = logging.getLogger('weight_logger')
        self.args_logger = logging.getLogger('args_logger')
        self.loss_logger = logging.getLogger('loss_logger')
        self.log_logger = logging.getLogger('log_logger')
        self.adjust_logger = logging.getLogger('adjust_logger')

        # 根据全局日志级别设置日志记录器的有效级别
        self._set_global_level()

    def _set_global_level(self):
        if self.global_level != 'DEBUG':
            # 只有结果日志生效
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