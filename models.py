from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertModel


def get_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_pythia_model(size: str):
    assert size in ["14m", "31m", "70m", "160m", "410m", "1b", "2.8b", "6.9b", "12b"]
    model_name = f"EleutherAI/pythia-{size}"
    return get_model(model_name)

def get_gpt2_model(size):
    assert size in ["small", "medium", "large", "xl"]
    if size == "small":
        model_name = "gpt2"
    else:
        model_name = f"gpt2-{size}"
    return get_model(model_name)

def get_bert_model(model_name: str):
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_model_from_name(model_name: str):
    """ Returns a model and its tokenizer
    """
    if "pythia" in model_name:
        return get_pythia_model(model_name.replace("pythia-", ""))
    if "gpt2" in model_name:
        return get_gpt2_model(model_name.replace("gpt2-", ''))
    if "bert" in model_name:
        return get_bert_model(model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}.")
