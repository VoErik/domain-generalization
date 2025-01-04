import argparse
from domgen.tuning import ParamTuner
from domgen.tuning import AugmentationTuner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config', type=str,
                        default='config/base/resnet18-pretrained/pacs-resnet18-pretrainedT-td=0.yaml', required=True)
    parser.add_argument('--hp_config', type=str, default='config/hp_search_space-2.yaml', required=True)
    parser.add_argument('--mode', type=str, choices=['hp', 'augment'], default='hp', required=True)
    args = parser.parse_args()

    if args.mode == 'hp':
        tuner = ParamTuner(args.base_config, args.hp_config)
    elif args.mode == 'augment':
        tuner = AugmentationTuner(args.base_config, args.hp_config)
    else:
        raise ValueError('Invalid mode. Choose from ["hp", "augment"]')
    best = tuner.run()
    print(best)




