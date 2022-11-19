from torchvision import transforms

# IMAGES
# mean: [0.06, 0.12, 0.08]
# std:  [0.04, 0.04, 0.05]

# resnet34
# mean: [0.485, 0.456, 0.406]
# std:  [0.229, 0.224, 0.225]

# densenet121 --> TOO BIG
# mean: [0.485, 0.456, 0.406]
# std:  [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                                    transforms.RandomVerticalFlip(0.5),
                                    transforms.RandomHorizontalFlip(0.5),
                                    #transforms.RandomEqualize(),
                                    #transforms.RandomAutocontrast(),
                                    transforms.Resize(224),
                                    #transforms.RandomCrop(224),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])

valid_transforms = transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])

test_transforms = transforms.Compose([
                                transforms.Resize(224),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
