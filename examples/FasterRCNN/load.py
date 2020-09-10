import tensorflow as tf

def load_session(pb_path):
    g = tf.Graph().as_default()
    output_graph_def = tf.compat.v1.GraphDef()
    with open(pb_path, "rb") as f:
        output_graph_def.ParseFromString(f.read()) 
    tf.import_graph_def(output_graph_def, name="")
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    sess = tf.compat.v1.Session(config=config)        
#     sess.run(tf.compat.v1.global_variables_initializer())

    input_image_tensor = sess.graph.get_tensor_by_name("image:0")
    output_tensor_boxes = sess.graph.get_tensor_by_name("output/boxes:0")
    output_tensor_scores = sess.graph.get_tensor_by_name("output/scores:0")
    output_tensor_labels = sess.graph.get_tensor_by_name("output/labels:0")
    output_tensor_masks = sess.graph.get_tensor_by_name("output/masks:0")

    output_tensors = [output_tensor_boxes, output_tensor_scores, output_tensor_labels, output_tensor_masks]

    return sess, input_image_tensor, output_tensors
