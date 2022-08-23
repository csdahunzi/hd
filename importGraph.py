import tensorflow as tf

class ImportGraph():
    """  Importing and running isolated TF graph """
    
    def __init__(self, loc,checkpoint):
        
        # Create local graph and use it in the session        
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint))
            
            # There are TWO options how to get activation operation:
            # FROM SAVED COLLECTION:            
            #self.activation = tf.get_collection('activation')[0]
            # BY NAME:
            self.op_to_restore = self.graph.get_tensor_by_name("prediction:0")
            self.op_to_restore1 = self.graph.get_tensor_by_name("gradient:0")

    def run(self, data):
        """ Running the activation operation previously imported """
        # The 'x' corresponds to name of input placeholder
        y = self.sess.run(self.op_to_restore, feed_dict={"X:0": data})
        dy = self.sess.run(self.op_to_restore1, feed_dict={"X:0": data})

        return y, dy
