import tensorflow as tf





class Weighted_Loss(tf.keras.metrics.Metric):
    
    def __init__(self, **kwargs):
        super(Weighted_Loss, self).__init__(**kwargs)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.sum_losses = self.add_weight("sum_losses", initializer="zeros")
        self.sum_weights = self.add_weight("sum_weights", initializer="zeros")
    

    def update_state(self, y_true, y_pred, sample_weight=None):
        sw = tf.cast(y_pred._keras_mask, dtype=y_pred.dtype)
        if sample_weight is not None:
            sw *= sample_weight
        losses = self.loss_object(y_true, y_pred, sw)
        self.sum_losses.assign_add(tf.reduce_sum(losses))
        self.sum_weights.assign_add(tf.reduce_sum(tf.cast(sw, dtype=losses.dtype)))


    def result(self):
        return self.sum_losses / self.sum_weights

 



    
class Factor(tf.keras.metrics.Metric):
    
    def __init__(self, **kwargs):
        super(Factor, self).__init__(**kwargs)
        self.sum_weights = self.add_weight("sum_weights", initializer="zeros")
        self.sum_samples = self.add_weight("sum_samples", initializer="zeros")
    

    def update_state(self, y_true, y_pred, sample_weight=None):
        sw = tf.cast(y_pred._keras_mask, dtype=y_pred.dtype)
        if sample_weight is not None:
            sw *= sample_weight
        self.sum_weights.assign_add(tf.reduce_sum(tf.cast(sw, dtype=tf.float32)))
        self.sum_samples.assign_add(tf.cast(tf.size(y_true), dtype=tf.float32))
   

    def result(self):
        return self.sum_weights / self.sum_samples
