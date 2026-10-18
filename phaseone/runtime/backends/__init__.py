"""Native providers. Imports stay inside worker processes until a model is loaded."""


def create(package, variant, cache_dir):
    if variant.backend == "mlx":
        from .mlx import MLXChat

        return MLXChat(package, variant, cache_dir)
    if variant.backend == "local_server":
        from .server import LocalServerChat

        return LocalServerChat(package, variant, cache_dir)
    if variant.engine == "llm":
        if variant.backend == "executorch" and variant.chat.runner == "qnn":
            from .qnn import QnnChat

            return QnnChat(package, variant, cache_dir)
        if variant.backend == "executorch" and variant.chat and variant.chat.processor:
            if variant.chat.processor.runner == "gemma4-ple":
                from .gemma4 import ExecuTorchGemma4

                return ExecuTorchGemma4(package, variant, cache_dir)
            from .multimodal import ExecuTorchMultimodal

            return ExecuTorchMultimodal(package, variant, cache_dir)
        from .chat import ExecuTorchChat, LiteRTChat

        provider = ExecuTorchChat if variant.backend == "executorch" else LiteRTChat
        return provider(package, variant, cache_dir)
    if variant.backend == "executorch":
        from .executorch import ExecuTorchTensor

        return ExecuTorchTensor(package, variant)
    from .litert import LiteRTTensor

    return LiteRTTensor(package, variant)
