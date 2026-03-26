import torch
import torchaudio


@torch.no_grad()
def decode_wave_vocoder(response, vocoder, codebook_sizes: list[int], audio_tokenizer):
    # 提取各个sample的音频长度
    response_len = (response[:,:,0] == codebook_sizes[0]).long().argmax(dim=1)
    valid_response_list = [response[i, :response_len[i], :] for i in range(response.shape[0]) if int(response_len[i])>0]
    if len(valid_response_list)==0:
        print("no valid response")
        return []
    flatten_response = torch.cat(valid_response_list, dim=0) if len(valid_response_list)>1 else valid_response_list[0]
    valid_response_len = response_len[response_len>0]
    ret = audio_tokenizer.decode(flatten_response.view(-1,response.shape[-1]),
                bridge_length=valid_response_len)
    batch_size = response.shape[0]
    valid_start = 0
    r = []
    for i in range(batch_size):
        if response_len[i]==0:
            r.append(None)
            continue
        if isinstance(ret, torch.Tensor):
            r.append(ret[valid_start:valid_start+1])
            valid_start+=1
            continue
        decode_wave = vocoder.decode(ret.flow_matching_mel[valid_start ][:ret.flow_matching_mel_lengths[valid_start ], :].transpose(0, 1).to(torch.float32).unsqueeze(0))
        r.append(decode_wave.cpu()) 
        valid_start+=1 
    return r

@torch.no_grad()
def decode_save_concat(response_list, vocoder, audio_tokenizer, codebook_sizes: list[int], path, sampling_rate=16000, wave_concat_overlap=800):
    wave_list = []
    for response in response_list: 
        wave_list.extend([wave_i for wave_i in decode_wave_vocoder(response, vocoder, audio_tokenizer=audio_tokenizer, codebook_sizes=codebook_sizes) if wave_i is not None])
    new_wave_list = [wave_list[0]]
    for w in wave_list[1:]:
        if new_wave_list[-1].shape[1] > wave_concat_overlap and w.shape[1] > wave_concat_overlap:
            new_wave_list.append((new_wave_list[-1][:, -wave_concat_overlap:] * torch.linspace(1.0, 0.0, wave_concat_overlap, device=new_wave_list[-1].device)[None, :] 
                                + w[:, :wave_concat_overlap] * torch.linspace(0.0, 1.0, wave_concat_overlap, device=new_wave_list[-1].device)[None, :]))
        new_wave_list.append(w)
    full_wave = torch.cat(new_wave_list, dim=1) if len(new_wave_list) > 1 else new_wave_list[0]
    torchaudio.save(path, full_wave, sampling_rate)  
