package server

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// toAPILogprobs converts llm.Logprobs to api.Logprobs
func toAPILogprobs(logprobs []llm.Logprob) []api.Logprob {
	result := make([]api.Logprob, len(logprobs))
	for i, lp := range logprobs {
		result[i] = api.Logprob{
			TokenLogprob: toAPITokenLogprob(lp.TokenLogprob),
		}
		if len(lp.TopLogprobs) > 0 {
			result[i].TopLogprobs = make([]api.TokenLogprob, len(lp.TopLogprobs))
			for j, tlp := range lp.TopLogprobs {
				result[i].TopLogprobs[j] = toAPITokenLogprob(tlp)
			}
		}
	}
	return result
}

func toAPITokenLogprob(token llm.TokenLogprob) api.TokenLogprob {
	raw := token.Bytes
	if raw == nil {
		raw = stringToByteInts(token.Token)
	}
	return api.TokenLogprob{Token: token.Token, Logprob: token.Logprob, Bytes: raw}
}

func stringToByteInts(s string) []int {
	if s == "" {
		return nil
	}

	raw := []byte(s)
	ints := make([]int, len(raw))
	for i, b := range raw {
		ints[i] = int(b)
	}
	return ints
}
