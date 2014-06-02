
# Learners: Tutorial

The `learners` module is organized arround the notion of channels. A `Channel`
has a name, describes a single scalar and can incorportate bounds. Here we
describe three channels, *x*, *y* and *a*, with bounds `[0, 10]` for the first
two and `[0, 100]` for *a*.


    import forest
    from learners import Channel, RandomLearner
    
    ch_x = Channel('x', [0, 10])
    ch_y = Channel('y', [0, 10])
    ch_a = Channel('a', [0, 100])

Then we create a learner that accept *x* and *y* as motor input, and learns
their mapping to sensory channel *a*. We create the configuration and
instanciate a `RandomLearner` instance, that return a random prediction.


    cfg = {'m_channels': [ch_x, ch_y],
           's_channels': [ch_a]}
    
    learner = RandomLearner(cfg)

We can then update the learner with an observation, a pair of motor and sensory
signal.


    learner.update({'x': 5, 'y': 4}, {'a': 9})

We can then ask the learner to predict the result of a motor signal.


    learner.predict({'x': 3, 'y': 4})




    {'a': 14.046620079802707}



Or infer the motor command that should produce a certain sensory output.


    learner.infer({'a': 3})




    {'x': 3.9660529699286418, 'y': 3.132155464083369}



Here the learner is returning random signal, and thus is particularly useless.
