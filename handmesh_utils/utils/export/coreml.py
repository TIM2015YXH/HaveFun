#!/usr/bin/env python
# coding=utf-8

import torch
import numpy as np
import onnxruntime
import os
from onnxruntime.datasets import get_example
from collections import OrderedDict
import cv2
from options.base_options import BaseOptions
from utils import spiral_tramsform
from utils.export.models.cmrpng_reg2d_left import CMRPNG_Reg2d_Left

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def run_onnx(input_images, onnx_name="convert.onnx"):
    onnx_path = "{}/{}".format(os.getcwd(), onnx_name)
    example_model = get_example(onnx_path)  # 一定要写绝对路径
    sess = onnxruntime.InferenceSession(example_model)

    if isinstance(input_images, list):
        # 多输入，input为dict
        inputs = OrderedDict()
        for i, input_image in enumerate(input_images):
            inputs[sess.get_inputs()[i].name] =  to_numpy(input_image)
    else:
        inputs = {sess.get_inputs()[0].name: to_numpy(input_images)}
    onnx_out = sess.run(None, inputs)

    # 输出为list，默认多输出
    return onnx_out

def run_pytorch(input_images, model):
    model.eval()
    # 多输入
    if isinstance(input_images, list):
        pytorch_out = model(*input_images)
    else:
        pytorch_out = model(input_images)

    # 多输出
    if isinstance(pytorch_out, tuple):
        pytorch_out = list(pytorch_out)
    return pytorch_out


def pytorch2onnx(model, input_images, name="convert", verify=True, process_output=None):
    model.eval()

    # export onnx model
    onnx_name = "{}.onnx".format(name)
    torch.onnx.export(model, input_images, onnx_name, input_names=["input"], verbose=True, output_names=["output"])

    if isinstance(input_images, tuple) :
        input_images = list(input_images)
    pytorch_out = run_pytorch(input_images, model)
    onnx_out = run_onnx(input_images, onnx_name)


    if process_output:
        print("PROCESS pytorch and onnx ouput ...")
        if isinstance(pytorch_out, list):
            np_pytorch_out = [to_numpy(out) for out in pytorch_out]
            process_output(np_pytorch_out, name="out_pytorch.png")
            process_output(onnx_out, name="out_onnx.png")
        else:
            process_output(to_numpy(pytorch_out), name="out_pytorch.png")
            process_output(onnx_out[0], name="out_onnx.png")

    if verify:
        # verify pytorch model and onnx model
        print("VERIFY pytorch and onnx model ...")
        if isinstance(pytorch_out, list):
            np_pytorch_out = [to_numpy(out) for out in pytorch_out]
            for index in range(0, len(np_pytorch_out)):
                np.testing.assert_almost_equal(np_pytorch_out[index], onnx_out[index], decimal=3)
                print(" verify {} output ok !".format(index))
        else:
            np.testing.assert_almost_equal(to_numpy(pytorch_out), onnx_out[0], decimal=3)
            print(" verify 0 output ok !")

    return pytorch_out, onnx_out


def run_mlmodel(input_images, coreml_name="convert.mlmodel"):
    import coremltools
    ml_path = "{}/{}".format(os.getcwd(), coreml_name)
    mlmodel = coremltools.models.MLModel(ml_path)
    spec = coremltools.utils.load_spec(ml_path)

    if isinstance(input_images, list):
        inputs = OrderedDict()
        # input_image.shape = 1,3,5
        for i, input_image in enumerate(input_images):
            input_name = spec.description.input[i].name
            inputs[input_name] = to_numpy(input_image).squeeze()
    else:
        inputs = {spec.description.input[0].name: to_numpy(input_images).squeeze(0)}

    #import pdb;pdb.set_trace()
    ml_out = mlmodel.predict(inputs, useCPUOnly=True)

    mlcore_out = []
    for i in range(len(spec.description.output)):
        mlcore_out.append(ml_out[spec.description.output[i].name])
    return mlcore_out


def pytorch2coreml(model, input_images, name = "convert", verify=True, process_output=None):
    model.eval()
    # export onnx model
    pytorch_out, onnx_out = pytorch2onnx(model, input_images, name, verify=False, process_output=None)

    if isinstance(input_images, tuple) :
        input_images = list(input_images)

    # export coreml model
    onnx_name = "{}.onnx".format(name)
    coreml_name = "{}.mlmodel".format(name)
    onnx_path = "{}/{}".format(os.getcwd(), onnx_name)

    from onnx_coreml import convert
    from onnxsim import simplify
    import onnx

    onnx_model = onnx.load(onnx_path)


    model_simple, check = simplify(onnx_model)


    #model_simple = onnx_model

    import pdb
    #pdb.set_trace()
    mlmodel = convert(model_simple)

    mlmodel.save(coreml_name)

    #input_images = input_images.squeeze(0)

    #pdb.set_trace()

    coreml_out = run_mlmodel(input_images, coreml_name)

    if process_output:
        print("PROCESS pytorch, onnx, coreml ouput ...")
        if isinstance(pytorch_out, list):
            np_pytorch_out = [to_numpy(out) for out in pytorch_out]
            process_output(np_pytorch_out, name="out_pytorch.png")
            process_output(onnx_out, name="out_onnx.png")
            process_output(coreml_out, name="out_coreml.png")
        else:
            process_output(pytorch_out, name="out_pytorch.png")
            process_output(onnx_out[0], name="out_onnx.png")
            process_output(coreml_out[0], name="out_coreml.png")

    if verify:
        # verify onnx model and mlmodel
        print("VERIFY pytorch, onnx, coreml model ...")
        if isinstance(pytorch_out, list):
            np_pytorch_out = [to_numpy(out) for out in pytorch_out]
            for index in range(0, len(np_pytorch_out)):
                np.testing.assert_almost_equal(np_pytorch_out[index].squeeze(), onnx_out[index].squeeze(), decimal=3)
                np.testing.assert_almost_equal(onnx_out[index].squeeze(), coreml_out[index].squeeze(), decimal=3)
                print(" verify {} output ok !".format(index))
        else:
            #import pdb;pdb.set_trace()
            np.testing.assert_almost_equal(to_numpy(pytorch_out['semi']).squeeze(), onnx_out[0].squeeze(), decimal=3)
            np.testing.assert_almost_equal(onnx_out[0].squeeze(), coreml_out[0].squeeze(), decimal=3)
            print(" verify 0 output ok !")

    print("compile mlmodelc file ...")
    os.system("xcrun coremlc compile {} . ".format(coreml_name))


if __name__ == '__main__':

    args = BaseOptions().parse()
    args.seq_length = [9, 9, 9, 9]
    args.out_channels = [32, 64, 128, 256]
    template_fp = os.path.join('../..', 'template', 'template.ply')
    transform_fp = os.path.join('../..', 'template', 'transform.pkl')
    spiral_indices_list, down_transform_list, up_transform_list, tmp = spiral_tramsform(transform_fp, template_fp, args.ds_factors, args.seq_length, args.dilation, None, 'cpu')
    for i in range(len(up_transform_list)):
        up_transform_list[i] = (*up_transform_list[i]._indices(), up_transform_list[i]._values())
    model = CMRPNG_Reg2d_Left(args, spiral_indices_list, up_transform_list)
    checkpoint = torch.load('/Users/chenxingyu/Documents/hand_mesh/lighthand/out/Kwai2D/cmrpng_reg2d_left_conm1cent_8gpu_fixbnparam_lr3/checkpoints/checkpoint_last.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    image = cv2.imread('/Users/chenxingyu/Tools/common/models/lighthand_reg2d_left/imgs/IMG_input_fist.png')
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)[..., ::-1]
    # image = np.expand_dims(image, 2)
    image = image.transpose(2, 0, 1)
    image = (image/255.-0.5)/0.5
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0)
    traced_model = torch.jit.trace(model, image)
    #preprocessing_args= {'image_scale' : (1.0/255.0)}
    import coremltools as ct
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_name", shape=image.shape)], #name "input_1" is used in 'quickstart'
        #**preprocessing_args,
        #classifier_config = ct.ClassifierConfig(class_labels) # provide only if step 2 was performed
        #classifier_config = preprocessing_args # provide only if step 2 was performed
    )
    mlmodel.save("cmrpng_reg2d_left.mlmodel")






