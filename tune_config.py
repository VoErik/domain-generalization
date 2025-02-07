import argparse
from domgen.tuning import ParamTuner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config', type=str,
                        default='assets/config/hy-tuning/base-models/resnet18-pretrained/pacs-resnet18-pretrainedT-td=0.yaml')
    parser.add_argument('--tune_config', type=str, default='assets/config/hp-tuning/hp_search_spaces/hp_search_space-2.yaml')
    parser.add_argument('--mode', type=str, choices=['hp', 'augment'], default='hp')
    args = parser.parse_args()

    if args.mode == 'hp':
        base = args.base_config
        tune = args.tune_config
        tuner = ParamTuner(base_config=base, tune_config=tune, mode='hp')
    elif args.mode == 'augment':
        raise NotImplementedError("Mode currently unavailable.")
        # tuner = AugmentationTuner(args.base_config, args.hp_config)
    else:
        raise ValueError('Invalid mode. Choose from ["hp", "augment"]')
    best = tuner.run()
    print(best)




