from utils import load, data_generator, model_selector, initialize_models, train, compute_nb_errors, plotacc, plotLoss
import argparse
import sys
import torch


parser = argparse.ArgumentParser()


parser.add_argument("model", help="one of the following: FNN, FNN_WS, FNN_AUX, FNN_WS_AUX, CNN, CNN_WS, CNN_AUX, CNN_WS_AUX")

parser.add_argument("--plot", help="plot the loss and accuracy curves", action="store_true")

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

torch.manual_seed(8)

train_input1, train_input2, test_input1, test_input2, train_classes1, train_classes2, test_classes1, test_classes2, train_target, test_target = load()
model_FNN, model_FNN_WS, model_FNN_AUX, model_FNN_WS_AUX, model_CNN, model_CNN_WS, model_CNN_AUX, model_CNN_WS_AUX = initialize_models()

if args.model.upper() == 'FNN':
	history = train(model_FNN, train_input1, train_input2, train_classes1,  train_classes2,train_target, \
                        test_input1, test_input2, test_classes1, test_classes2,  test_target)
elif args.model.upper() == 'FNN_WS':
	history = train(model_FNN_WS, train_input1, train_input2, train_classes1,  train_classes2,train_target, \
                        test_input1, test_input2, test_classes1, test_classes2,  test_target)
elif args.model.upper() == 'FNN_AUX':
	history = train(model_FNN_AUX, train_input1, train_input2, train_classes1,  train_classes2,train_target, \
                        test_input1, test_input2, test_classes1, test_classes2,  test_target)
elif args.model.upper() == 'FNN_WS_AUX':
	history = train(model_FNN_AUX, train_input1, train_input2, train_classes1,  train_classes2,train_target, \
                        test_input1, test_input2, test_classes1, test_classes2,  test_target)
elif args.model.upper() == 'CNN':
	history = train(model_CNN, train_input1, train_input2, train_classes1,  train_classes2,train_target, \
                        test_input1, test_input2, test_classes1, test_classes2,  test_target)
elif args.model.upper() == 'CNN_WS':
	history = train(model_CNN_WS, train_input1, train_input2, train_classes1,  train_classes2,train_target, \
                        test_input1, test_input2, test_classes1, test_classes2,  test_target)
elif args.model.upper() == 'CNN_AUX':
	history = train(model_CNN_AUX, train_input1, train_input2, train_classes1,  train_classes2,train_target, \
                        test_input1, test_input2, test_classes1, test_classes2,  test_target)
elif args.model.upper() == 'CNN_WS_AUX':
	history = train(model_CNN_WS_AUX, train_input1, train_input2, train_classes1,  train_classes2,train_target, \
                        test_input1, test_input2, test_classes1, test_classes2,  test_target)


if args.plot:
    plotacc([history],  args.model.upper())
    plotLoss([history], args.model.upper())

print(args.model.upper())



