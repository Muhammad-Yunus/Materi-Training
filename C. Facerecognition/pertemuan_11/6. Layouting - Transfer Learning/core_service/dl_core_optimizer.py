import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.tools import optimize_for_inference_lib

class ModelOptimizer():
    def h5_to_savedModel(self, model_name='model-cnn-facerecognition.h5', savedModel_folder="tf_model"):
        model = tf.keras.models.load_model(model_name)
        model.save(savedModel_folder)
            
    def optimize(self, savedModel_folder="tf_model", target_name='frozen_graph.pb'):
        importer = tf.saved_model.load(savedModel_folder)
        infer = importer.signatures['serving_default']
        f = convert_variables_to_constants_v2(infer)
        graph_def = f.graph.as_graph_def()

        input_name =  graph_def.node[0].name
        output_name =  graph_def.node[-1].name
        
        f = convert_variables_to_constants_v2(infer)
        graph_def = f.graph.as_graph_def()

        graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def,
                                                                      [input_name],
                                                                      [output_name],
                                                                      tf.float32.as_datatype_enum)

        with tf.io.gfile.GFile(target_name, 'wb') as f:
            f.write(graph_def.SerializeToString())