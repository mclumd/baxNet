# Description:
# Example TensorFlow models for ImageNet.

package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = ["//baxNet/..."],
)

py_library(
    name = "dataset",
    srcs = [
        "dataset.py",
    ],
)

py_library(
    name = "imagenet_data",
    srcs = [
        "imagenet_data.py",
    ],
    deps = [
        ":dataset",
    ],
)

py_library(
    name = "image_processing",
    srcs = [
        "image_processing.py",
    ],
)

py_library(
    name = "baxNet",
    srcs = [
        "baxNet_model.py",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "baxNet_eval",
    srcs = [
        "baxNet_eval.py",
    ],
    deps = [
        ":image_processing",
        ":baxNet",
    ],
)

py_library(
    name = "baxNet_multi_gpu_train",
    srcs = [
        "baxNet_multi_gpu_train.py",
    ],
    deps = [
        ":image_processing",
	":imagenet_data",
        ":baxNet",
    ],
)

py_binary(
    name = "build_image_data",
    srcs = ["data/build_image_data.py"],
)

sh_binary(
    name = "download_and_preprocess_imagenet",
    srcs = ["data/download_and_preprocess_imagenet.sh"],
    data = [
        "data/download_imagenet.sh",
        "data/imagenet_2012_validation_synset_labels.txt",
        "data/imagenet_lsvrc_2015_synsets.txt",
        "data/imagenet_metadata.txt",
        "data/preprocess_imagenet_validation_data.py",
        "data/process_bounding_boxes.py",
        ":build_imagenet_data",
    ],
)

py_binary(
    name = "build_imagenet_data",
    srcs = ["data/build_imagenet_data.py"],
)

filegroup(
    name = "srcs",
    srcs = glob(
        [
            "**/*.py",
            "BUILD",
        ],
    ),
)

filegroup(
    name = "imagenet_metadata",
    srcs = [
        "data/imagenet_lsvrc_2015_synsets.txt",
        "data/imagenet_metadata.txt",
    ],
    visibility = ["//visibility:public"],
)
