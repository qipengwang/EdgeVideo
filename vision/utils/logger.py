import logging
import os

class Logger():
    logger = None
    log_name = 'default'
    
    @classmethod
    def get_logger(cls):
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if Logger.logger is not None:
            return Logger.logger
        log_file = 'log/{}.log'.format(Logger.log_name)
        logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format=fmt)
        Logger.logger = logging.getLogger('EdgeVideo')
        sh = logging.StreamHandler()    #往屏幕上输出
        sh.setFormatter(logging.Formatter(fmt))
        Logger.logger.addHandler(sh)
        Logger.logger.info('logger init finished ---- log file: {}'.format(log_file))
        return Logger.logger
    
    @classmethod
    def set_log_name(cls, name):
        name = os.path.basename(name)
        while name[-4:] == '.log':
            name = name[:-4]
        Logger.log_name = name