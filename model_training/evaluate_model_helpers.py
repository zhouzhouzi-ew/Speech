import torch
import numpy as np
import h5py
import time
import re
import json
from pathlib import Path
from typing import Sequence

from data_augmentations import gauss_smooth

LOGIT_TO_PHONEME = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ',
]

_PHONEME_ALIASES = {
    '<blank>': 'BLANK',
    '<sil>': ' | ',
    'SIL': ' | ',
    '|': ' | ',
}


def normalize_phoneme_label(label):
    return _PHONEME_ALIASES.get(label, label)


def strip_phoneme_stress(label):
    return re.sub(r'[0-9]', '', label)


def decode_text_value(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.bytes_):
        return value.decode('utf-8')
    if isinstance(value, np.ndarray):
        if value.dtype.kind == 'S':
            return b''.join(value.tolist()).decode('utf-8')
        if value.dtype.kind == 'U':
            return ''.join(value.tolist())
    return str(value)


def load_session_phoneme_order(metadata_path):
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    labels = metadata.get('labels', metadata)
    phoneme_to_id = labels.get('phoneme_to_id')
    if phoneme_to_id is None:
        raise ValueError(f'Could not find phoneme_to_id in {metadata_path}')
    ordered = [phoneme for phoneme, _ in sorted(phoneme_to_id.items(), key=lambda item: item[1])]
    return [normalize_phoneme_label(phoneme) for phoneme in ordered]


def load_session_metadata(metadata_path):
    return json.loads(Path(metadata_path).read_text(encoding='utf-8'))


def expand_logits_to_official_order(logits, source_order: Sequence[str], target_order: Sequence[str] = LOGIT_TO_PHONEME,
                                    missing_fill: float = -1e4):
    arr = np.asarray(logits)
    source_order = [normalize_phoneme_label(phoneme) for phoneme in source_order]
    target_order = [normalize_phoneme_label(phoneme) for phoneme in target_order]

    if arr.shape[-1] != len(source_order):
        raise ValueError(
            f'logits last dimension ({arr.shape[-1]}) does not match source_order length ({len(source_order)})'
        )

    if arr.shape[-1] == len(target_order) and source_order == target_order:
        return arr.copy()

    source_index = {phoneme: idx for idx, phoneme in enumerate(source_order)}
    expanded = np.full(arr.shape[:-1] + (len(target_order),), missing_fill, dtype=arr.dtype)
    for target_idx, phoneme in enumerate(target_order):
        source_idx = source_index.get(phoneme)
        if source_idx is not None:
            expanded[..., target_idx] = arr[..., source_idx]
    return expanded


def load_phoneme_word_lexicon(dict_path):
    lexicon = {}
    with open(dict_path, 'r', encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(';;;'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            phonemes = tuple(normalize_phoneme_label(strip_phoneme_stress(p)) for p in parts[1:])
            if phonemes not in lexicon:
                lexicon[phonemes] = word
    return lexicon


def phonemes_to_words(phonemes, phoneme_word_lexicon, unknown_token='<unk>'):
    words = []
    chunk = []
    for phoneme in phonemes:
        phoneme = normalize_phoneme_label(strip_phoneme_stress(str(phoneme)))
        if phoneme == 'BLANK':
            continue
        if phoneme == ' | ':
            if chunk:
                words.append(phoneme_word_lexicon.get(tuple(chunk), unknown_token))
                chunk = []
            continue
        chunk.append(phoneme)
    if chunk:
        words.append(phoneme_word_lexicon.get(tuple(chunk), unknown_token))
    return words


def sequence_edit_distance(reference, hypothesis):
    prev = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        cur = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            cost = 0 if ref_item == hyp_item else 1
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = cur
    return prev[-1]

def _extract_transcription(input):
    endIdx = np.argwhere(input == 0)[0, 0]
    trans = ''
    for c in range(endIdx):
        trans += chr(input[c])
    return trans

def load_h5py_file(file_path, b2txt_csv_df):
    data = {
        'neural_features': [],
        'n_time_steps': [],
        'seq_class_ids': [],
        'seq_len': [],
        'transcriptions': [],
        'sentence_label': [],
        'subject': [],
        'date': [],
        'session': [],
        'raw_session': [],
        'paired_diagnostic_session': [],
        'paired_diagnostic_block_num': [],
        'block_num': [],
        'trial_num': [],
        'split': [],
        'corpus': [],
    }
    # Open the hdf5 file for that day
    with h5py.File(file_path, 'r') as f:

        keys = list(f.keys())

        # For each trial in the selected trials in that day
        for key in keys:
            g = f[key]

            neural_features = g['input_features'][:]
            n_time_steps = g.attrs['n_time_steps']
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = decode_text_value(g.attrs['sentence_label']) if 'sentence_label' in g.attrs else None
            session = decode_text_value(g.attrs['session'])
            raw_session = decode_text_value(g.attrs['raw_session']) if 'raw_session' in g.attrs else session
            paired_diagnostic_session = decode_text_value(g.attrs['paired_diagnostic_session']) if 'paired_diagnostic_session' in g.attrs else None
            paired_diagnostic_block_num = int(g.attrs['paired_diagnostic_block_num']) if 'paired_diagnostic_block_num' in g.attrs else None
            subject = decode_text_value(g.attrs['subject']) if 'subject' in g.attrs else None
            date = decode_text_value(g.attrs['date']) if 'date' in g.attrs else None
            split = decode_text_value(g.attrs['split']) if 'split' in g.attrs else None
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']

            if date is None:
                parts = session.split('.')
                if len(parts) >= 4:
                    date = f'{parts[1]}-{parts[2]}-{parts[3]}'
                else:
                    raise ValueError(f'Could not infer date from session attr {session!r} in {file_path}')

            if 'corpus' in g.attrs:
                corpus_name = decode_text_value(g.attrs['corpus'])
            else:
                if b2txt_csv_df is None:
                    raise ValueError(f'No corpus attr found in {file_path} and csv lookup is unavailable.')
                row = b2txt_csv_df[(b2txt_csv_df['Date'] == date) & (b2txt_csv_df['Block number'] == block_num)]
                if len(row) == 0:
                    raise ValueError(f'Could not find corpus for date={date}, block={block_num} in {file_path}')
                corpus_name = row['Corpus'].values[0]

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['subject'].append(subject)
            data['date'].append(date)
            data['session'].append(session)
            data['raw_session'].append(raw_session)
            data['paired_diagnostic_session'].append(paired_diagnostic_session)
            data['paired_diagnostic_block_num'].append(paired_diagnostic_block_num)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)
            data['split'].append(split)
            data['corpus'].append(corpus_name)
    return data

def rearrange_speech_logits_pt(logits):
    # original order is [BLANK, phonemes..., SIL]
    # rearrange so the order is [BLANK, SIL, phonemes...]
    logits = np.concatenate((logits[:, :, 0:1], logits[:, :, -1:], logits[:, :, 1:-1]), axis=-1)
    return logits

# single decoding step function.
# smooths data and puts it through the model.
def runSingleDecodingStep(x, input_layer, model, model_args, device):

    # Use autocast for efficiency
    use_cuda_amp = bool(model_args['use_amp']) and device.type == 'cuda'
    with torch.autocast(device_type = device.type, enabled = use_cuda_amp, dtype = torch.bfloat16):

        x = gauss_smooth(
            inputs = x, 
            device = device,
            smooth_kernel_std = model_args['dataset']['data_transforms']['smooth_kernel_std'],
            smooth_kernel_size = model_args['dataset']['data_transforms']['smooth_kernel_size'],
            padding = 'same',
        )

        with torch.no_grad():
            logits, _ = model(
                x = x,
                day_idx = torch.tensor([input_layer], device=device),
                states = None, # no initial states
                return_state = True,
            )

    # convert logits from bfloat16 to float32
    logits = logits.float().cpu().numpy()

    # # original order is [BLANK, phonemes..., SIL]
    # # rearrange so the order is [BLANK, SIL, phonemes...]
    # logits = rearrange_speech_logits_pt(logits)

    return logits

def remove_punctuation(sentence):
    # Remove punctuation
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()

    sentence = sentence.strip()
    sentence = ' '.join([word for word in sentence.split() if word != ''])

    return sentence

def get_current_redis_time_ms(redis_conn):
    t = redis_conn.time()
    return int(t[0]*1000 + t[1]/1000)


######### language model helper functions ##########

def reset_remote_language_model(
        r,
        remote_lm_done_resetting_lastEntrySeen,
    ):
    
    r.xadd('remote_lm_reset', {'done': 0})
    time.sleep(0.001)
    # print('Resetting remote language model before continuing...')
    remote_lm_done_resetting = []
    while len(remote_lm_done_resetting) == 0:
        remote_lm_done_resetting = r.xread(
            {'remote_lm_done_resetting': remote_lm_done_resetting_lastEntrySeen},
            count=1,
            block=10000,
        )
        if len(remote_lm_done_resetting) == 0:
            print(f'Still waiting for remote lm reset from ts {remote_lm_done_resetting_lastEntrySeen}...')
    for entry_id, entry_data in remote_lm_done_resetting[0][1]:
        remote_lm_done_resetting_lastEntrySeen = entry_id
        # print('Remote language model reset.')

    return remote_lm_done_resetting_lastEntrySeen


def update_remote_lm_params(
        r,
        remote_lm_done_updating_lastEntrySeen,
        acoustic_scale=0.35,
        blank_penalty=90.0,
        alpha=0.55,
    ):
    
    # update remote lm params
    entry_dict = {
        # 'max_active': max_active,
        # 'min_active': min_active,
        # 'beam': beam,
        # 'lattice_beam': lattice_beam,
        'acoustic_scale': acoustic_scale,
        # 'ctc_blank_skip_threshold': ctc_blank_skip_threshold,
        # 'length_penalty': length_penalty,
        # 'nbest': nbest,
        'blank_penalty': blank_penalty,
        'alpha': alpha,
        # 'do_opt': do_opt,
        # 'rescore': rescore,
        # 'top_candidates_to_augment': top_candidates_to_augment,
        # 'score_penalty_percent': score_penalty_percent,
        # 'specific_word_bias': specific_word_bias,
    }

    r.xadd('remote_lm_update_params', entry_dict)
    time.sleep(0.001)
    remote_lm_done_updating = []
    while len(remote_lm_done_updating) == 0:
        remote_lm_done_updating = r.xread(
            {'remote_lm_done_updating_params': remote_lm_done_updating_lastEntrySeen},
            block=10000,
            count=1,
        )
        if len(remote_lm_done_updating) == 0:
            print(f'Still waiting for remote lm to update parameters from ts {remote_lm_done_updating_lastEntrySeen}...')
    for entry_id, entry_data in remote_lm_done_updating[0][1]:
        remote_lm_done_updating_lastEntrySeen = entry_id
        # print('Remote language model params updated.')

    return remote_lm_done_updating_lastEntrySeen


def send_logits_to_remote_lm(
        r,
        remote_lm_input_stream,
        remote_lm_output_partial_stream,
        remote_lm_output_partial_lastEntrySeen,
        logits,
    ):
    
    # put logits into remote lm and get partial output
    r.xadd(remote_lm_input_stream, {'logits': np.float32(logits).tobytes()})
    remote_lm_output = []
    while len(remote_lm_output) == 0:
        remote_lm_output = r.xread(
            {remote_lm_output_partial_stream: remote_lm_output_partial_lastEntrySeen},
            block=10000,
            count=1,
        )
        if len(remote_lm_output) == 0:
            print(f'Still waiting for remote lm partial output from ts {remote_lm_output_partial_lastEntrySeen}...')
    for entry_id, entry_data in remote_lm_output[0][1]:
        remote_lm_output_partial_lastEntrySeen = entry_id
        decoded = entry_data[b'lm_response_partial'].decode()

    return remote_lm_output_partial_lastEntrySeen, decoded


def finalize_remote_lm(
        r,
        remote_lm_output_final_stream,
        remote_lm_output_final_lastEntrySeen,
    ):
    
    # finalize remote lm
    r.xadd('remote_lm_finalize', {'done': 0})
    time.sleep(0.005)
    remote_lm_output = []
    while len(remote_lm_output) == 0:
        remote_lm_output = r.xread(
            {remote_lm_output_final_stream: remote_lm_output_final_lastEntrySeen},
            block=10000,
            count=1,
        )
        if len(remote_lm_output) == 0:
            print(f'Still waiting for remote lm final output from ts {remote_lm_output_final_lastEntrySeen}...')
    # print('Received remote lm final output.')

    for entry_id, entry_data in remote_lm_output[0][1]:
        remote_lm_output_final_lastEntrySeen = entry_id

        candidate_sentences = [str(c) for c in entry_data[b'scoring'].decode().split(';')[::5]]
        candidate_acoustic_scores = [float(c) for c in entry_data[b'scoring'].decode().split(';')[1::5]]
        candidate_ngram_scores = [float(c) for c in entry_data[b'scoring'].decode().split(';')[2::5]]
        candidate_llm_scores = [float(c) for c in entry_data[b'scoring'].decode().split(';')[3::5]]
        candidate_total_scores = [float(c) for c in entry_data[b'scoring'].decode().split(';')[4::5]]


    # account for a weird edge case where there are no candidate sentences
    if len(candidate_sentences) == 0 or len(candidate_total_scores) == 0:
        print('No candidate sentences were received from the language model.')
        candidate_sentences = ['']
        candidate_acoustic_scores = [0]
        candidate_ngram_scores = [0]
        candidate_llm_scores = [0]
        candidate_total_scores = [0]

    else:
        # sort candidate sentences by total score (higher is better)
        sort_order = np.argsort(candidate_total_scores)[::-1]

        candidate_sentences = [candidate_sentences[i] for i in sort_order]
        candidate_acoustic_scores = [candidate_acoustic_scores[i] for i in sort_order]
        candidate_ngram_scores = [candidate_ngram_scores[i] for i in sort_order]
        candidate_llm_scores = [candidate_llm_scores[i] for i in sort_order]
        candidate_total_scores = [candidate_total_scores[i] for i in sort_order]

    # loop through candidates backwards and remove any duplicates
    for i in range(len(candidate_sentences)-1, 0, -1):
        if candidate_sentences[i] in candidate_sentences[:i]:
            candidate_sentences.pop(i)
            candidate_acoustic_scores.pop(i)
            candidate_ngram_scores.pop(i)
            candidate_llm_scores.pop(i)
            candidate_total_scores.pop(i)

    lm_out = {
        'candidate_sentences': candidate_sentences,
        'candidate_acoustic_scores': candidate_acoustic_scores,
        'candidate_ngram_scores': candidate_ngram_scores,
        'candidate_llm_scores': candidate_llm_scores,
        'candidate_total_scores': candidate_total_scores,
    }

    return remote_lm_output_final_lastEntrySeen, lm_out
