from multiprocessing import Process, Queue, Event
import os
import time

def traj_segment_function(pi, env, n_episodes, horizon, states, terminals):
    '''
    Collects trajectories
    '''

    # Initialize state variables
    rets = []
    for i, s in enumerate(states):
        if terminals[i]:
            rets.append(0)
            continue
        t = 0
        env.set_signature(s)
        done = False
        ret = 0
        while t < horizon and not done:
            action = pi(s)
            s, r, done, _ = env.step(action)
            t += 1
            ret += r
        rets.append(ret)
    return rets


def generate_next_particles(env, action, states, terminals, seeds):
    particles = []
    for i, s in enumerate(states):
        if terminals[i]:
            particles.append((s, seeds[i], 0, True))
            continue
        env.set_signature(s)
        s, r, done, _ = env.step(action)
        particles.append((env.get_signature(), seeds[i], r, done))
    return particles


class Worker(Process):
    '''
    A worker is an independent process with its own environment and policy instantiated locally
    after being created. It ***must*** be runned before creating any tensorflow session!
    '''

    def __init__(self, output, inp, event, make_env, make_pi, traj_segment_generator, step_function, n_episodes, seed):
        super(Worker, self).__init__()
        self.output = output
        self.input = inp
        self.make_env = make_env
        self.make_pi = make_pi
        self.traj_segment_generator = traj_segment_generator
        self.step_function = step_function
        self.event = event
        self.seed = seed
        self.n_episodes = n_episodes

    def run(self):

        env = self.make_env()
        workerseed = self.seed + 10000
        env.seed(workerseed)
        pi = self.make_pi(env.action_space)
        print('Worker %s - Running with seed %s' % (os.getpid(), workerseed))
        while True:
            self.event.wait()
            self.event.clear()
            command, args = self.input.get()
            if command == 'collect':
                horizon = args['horizon']
                terminals = args['terminals']
                states = args['states']
                # print('Worker %s - Collecting...' % os.getpid())
                rets = self.traj_segment_generator(pi, env, horizon, states, terminals)
                self.output.put((os.getpid(), rets))
            elif command == 'step':
                terminals = args['terminals']
                states = args['states']
                seeds = args['seeds']
                action = args['action']
                # print('Worker %s - Collecting...' % os.getpid())
                samples = self.step_function(env, action, states, terminals, seeds)
                self.output.put((os.getpid(), samples))
            elif command == 'exit':
                print('Worker %s - Exiting...' % os.getpid())
                #env.close()
                break


class ParallelSampler(object):

    def __init__(self, make_pi, make_env, n_particles,  n_workers=-1, seed=0):
        affinity = len(os.sched_getaffinity(0))
        if n_workers == -1:
            self.n_workers = affinity
        else:
            self.n_workers = min(n_workers, affinity)
        self.n_workers = min(n_particles, self.n_workers)
        print('Using %s CPUs' % self.n_workers)

        if seed is None:
            seed = time.time()

        self.output_queue = Queue()
        self.input_queues = [Queue() for _ in range(self.n_workers)]
        self.events = [Event() for _ in range(self.n_workers)]

        n_episodes_per_process = n_particles // self.n_workers
        remainder = n_particles % self.n_workers

        f = lambda pi, env, horizon, states, terminals: traj_segment_function(pi, env, n_episodes_per_process,
                                                                              horizon, states, terminals)
        f_rem = lambda pi, env, horizon, states, terminals: traj_segment_function(pi, env,
                                                                                  n_episodes_per_process + 1, horizon,
                                                                                  states, terminals)
        f_step = lambda env, action, states, terminals, seeds: generate_next_particles(env, action, states,
                                                                                       terminals, seeds)
        fun = [f] * (self.n_workers - remainder) + [f_rem] * remainder
        fun_steps = [f_step] * self.n_workers
        episodes = [n_episodes_per_process] * (self.n_workers - remainder) + [n_episodes_per_process + 1] * remainder
        self.workers = [Worker(output=self.output_queue,
                               inp=self.input_queues[i],
                               event=self.events[i],
                               make_env=make_env,
                               make_pi=make_pi,
                               traj_segment_generator=fun[i],
                               step_function=fun_steps[i],
                               n_episodes=episodes[i],
                               seed=seed + i) for i in range(self.n_workers)]

        for w in self.workers:
            w.start()

    def evaluate(self, particles, horizon=200):
        current = 0
        for i in range(self.n_workers):
            num_particles = self.workers[i].n_episodes
            current_particles = particles[current: current + num_particles]
            current += num_particles
            args = {'horizon': horizon,
                    'terminals': [p.terminal for p in current_particles],
                    'states': [p.state for p in current_particles]}

            self.input_queues[i].put(('collect', args))

        for e in self.events:
            e.set()

        returns = []
        for i in range(self.n_workers):
            pid, rets = self.output_queue.get()
            returns += rets

        return returns

    def generate_next_particles(self, particles, action):
        current = 0
        for i in range(self.n_workers):
            num_particles = self.workers[i].n_episodes
            current_particles = particles[current: current + num_particles]
            current += num_particles
            args = {
                    'terminals': [p.terminal for p in current_particles],
                    'states': [p.state for p in current_particles],
                    'seeds': [p.seed for p in current_particles],
                    'action': action}

            self.input_queues[i].put(('step', args))

        for e in self.events:
            e.set()

        particles = []
        for i in range(self.n_workers):
            pid, p = self.output_queue.get()
            particles += p

        return particles


    def close(self):
        for i in range(self.n_workers):
            self.input_queues[i].put(('exit', None))

        for e in self.events:
            e.set()

        for w in self.workers:
            w.join()
