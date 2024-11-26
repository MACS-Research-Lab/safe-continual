from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1

TASK_LENGTH = 1_000_000 / 10
# TASK_LENGTH = 350 / 10
# TASK_CYCLE = ['nominal', 'shorter', 'nominal', 'eo', 'shorter_eo', 'shorter']
TASK_CYCLE = ['nominal', 'left', 'nominal', 'right', 'left', 'right'] # recommend 7 million

class GoalLevel3(GoalLevel1):
    def __init__(self, config) -> None:
        self.steps_since_change = 0
        self.current_task_num = 0
        self.current_task_name = TASK_CYCLE[self.current_task_num]
        super().__init__(config=config)
        self.activate_task(self.current_task_name)

    def specific_step(self):
        self.steps_since_change += 1
        self.choose_task()
        return super().specific_step()
    
    def choose_task(self):
        if self.steps_since_change > TASK_LENGTH:
            self.current_task_num = (self.current_task_num + 1) % len(TASK_CYCLE)
            self.current_task_name = TASK_CYCLE[self.current_task_num]
            self.activate_task(self.current_task_name)
            self.steps_since_change = 0

    def activate_task(self, task_name):
        if task_name == 'nominal':
            return self.nominal_task()
        elif task_name == 'shorter':
            return self.shorter_lidar_task()
        elif task_name == 'eo':
            return self.every_other_lidar_faulty_task()
        elif task_name == 'shorter_eo':
            return self.shorter_everyother_task()
        elif task_name == 'left':
            return self.left_faulty_task()
        elif task_name == 'right':
            return self.right_faulty_task()
        else:
            raise Exception(f"Unsupported task {task_name} specified")

    def nominal_task(self):
        self.lidar_conf.max_dist = 3
        self.lidar_conf.off_indices = None

    def shorter_lidar_task(self):
        self.lidar_conf.max_dist = 1.5
        self.lidar_conf.off_indices = None

    def right_faulty_task(self):
        self.lidar_conf.max_dist = 3
        self.lidar_conf.off_indices = [False, False, True, True,
                                       True, True, True, True,
                                       True, True, True, True,
                                       True, True, False, False]
                                       

    def left_faulty_task(self):
        self.lidar_conf.max_dist = 3
        self.lidar_conf.off_indices = [True, True, True, True,
                                       True, True, False, False,
                                       False, False, True, True,
                                       True, True, True, True]

        
    def every_other_lidar_faulty_task(self):
        self.lidar_conf.max_dist = 3
        self.lidar_conf.off_indices = [True, False, True, False,
                                       True, False, True, False,
                                       True, False, True, False,
                                       True, False, True, False,]
        
    def shorter_everyother_task(self):
        self.lidar_conf.max_dist = 1.5
        self.lidar_conf.off_indices = [True, False, True, False,
                                       True, False, True, False,
                                       True, False, True, False,
                                       True, False, True, False,]