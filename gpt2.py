from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model

lang_pretrained = {
    'en': {
        'tokenizer': 'gpt2-medium',
        'lmmodel': 'gpt2-medium',
        'model': 'gpt2-medium'
    },
    'fr': {
        'tokenizer': 'asi/gpt-fr-cased-small',
        'lmmodel': 'asi/gpt-fr-cased-small',
        'model': 'asi/gpt-fr-cased-small'
    },
    'fi': {
        'tokenizer': 'Finnish-NLP/gpt2-medium-finnish',
        'lmmodel': 'Finnish-NLP/gpt2-medium-finnish',
        'model': 'Finnish-NLP/gpt2-medium-finnish'
    }
}

loaded_models = {
    'en': {},
    'fr': {},
    'fi': {},
    'zh': {}
}

def get_gpt2_lm(lang):
    if 'lmmodel' not in loaded_models[lang]:
        loaded_models[lang]['lmmodel'] = GPT2LMHeadModel.from_pretrained(
            lang_pretrained[lang]['lmmodel']).cuda()
        loaded_models[lang]['lmmodel'].eval()
    return loaded_models[lang]['lmmodel']

def get_gpt2_tokenizer(lang):
    if 'tokenizer' not in loaded_models[lang]:
        loaded_models[lang]['tokenizer'] = GPT2Tokenizer.from_pretrained(
            lang_pretrained[lang]['tokenizer'])
    return loaded_models[lang]['tokenizer']

# TODO zh with roformer instead.

from transformers import BertTokenizer, RoFormerForCausalLM, RoFormerConfig

def get_zh_tokenizer():
    if 'tokenizer' not in loaded_models['zh']:
        loaded_models['zh']['tokenizer'] = BertTokenizer.from_pretrained('junnyu/roformer_chinese_sim_char_base')
    return loaded_models['zh']['tokenizer']

def get_zh_lm():
    if 'lmmodel' not in loaded_models['zh']:
        tokenizer = get_zh_tokenizer()

        config = RoFormerConfig.from_pretrained('junnyu/roformer_chinese_sim_char_base')
        config.is_decoder = True
        config.eos_token_id = tokenizer.sep_token_id
        config.pooler_activation = 'linear'
        model = RoFormerForCausalLM.from_pretrained('junnyu/roformer_chinese_sim_char_base',
                config = config).cuda()

        loaded_models['zh']['lmmodel'] = model

    return loaded_models['zh']['lmmodel']
