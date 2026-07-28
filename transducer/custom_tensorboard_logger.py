"""TensorBoard logging with epoch-aligned steps for distributed training.

Only the main process writes events, preventing duplicate logs during
multi-GPU SpeechBrain experiments.
"""

from speechbrain.utils.distributed import if_main_process, main_process_only
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)



class TrainLogger:
    """Abstract class defining an interface for training loggers."""


    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=False,
    ):
        """Log the stats for one epoch.

        Arguments
        ---------
        stats_meta : dict of str:scalar pairs
            Meta information about the stats (e.g., epoch, learning-rate, etc.).
        train_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the training pass.
        valid_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the validation pass.
        test_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the test pass.
        verbose : bool
            Whether to also put logging information to the standard logger.
        """
        raise NotImplementedError





class TensorboardLogger(TrainLogger):
    """Logs training information in the format required by Tensorboard.

    Arguments
    ---------
    save_dir : str
        A directory for storing all the relevant logs.

    Raises
    ------
    ImportError if Tensorboard is not installed.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir

        # Raises ImportError if TensorBoard is not installed
        from torch.utils.tensorboard import SummaryWriter

        # Initialize writer only on main
        self.writer = None
        if if_main_process():
            self.writer = SummaryWriter(self.save_dir)
        self.global_step = {"train": {}, "valid": {}, "test": {}, "meta": 0}


    @main_process_only
    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=False,
    ):
        """See TrainLogger.log_stats()"""
        if "epoch" in stats_meta:
            epoch = int(stats_meta["epoch"])

            # Make meta align so parent increment puts it at `epoch`
            self.global_step["meta"] = epoch - 1

            # Also align per-stat counters so the next +1 makes them == epoch
            for dataset, stats in [("train", train_stats), ("valid", valid_stats), ("test", test_stats)]:
                if stats is None:
                    continue
                for stat in stats.keys():
                    # initialize if missing OR if it looks reset/smaller than epoch-1
                    cur = self.global_step[dataset].get(stat, -1)
                    if cur < epoch - 1:
                        self.global_step[dataset][stat] = epoch - 1
   
        self.global_step["meta"] += 1
        for name, value in stats_meta.items():
            self.writer.add_scalar(name, value, self.global_step["meta"])

        for dataset, stats in [
            ("train", train_stats),
            ("valid", valid_stats),
            ("test", test_stats),
        ]:
            if stats is None:
                continue
            for stat, value_list in stats.items():
                if stat not in self.global_step[dataset]:
                    self.global_step[dataset][stat] = 0
                tag = f"{stat}/{dataset}"

                # Both single value (per Epoch) and list (Per batch) logging is supported
                if isinstance(value_list, list):
                    for value in value_list:
                        new_global_step = self.global_step[dataset][stat] + 1
                        self.writer.add_scalar(tag, value, new_global_step)
                        self.global_step[dataset][stat] = new_global_step
                else:
                    value = value_list
                    new_global_step = self.global_step[dataset][stat] + 1
                    self.writer.add_scalar(tag, value, new_global_step)
                    self.global_step[dataset][stat] = new_global_step


