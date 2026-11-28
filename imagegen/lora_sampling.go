package imagegen

import (
	"math"
	"strconv"
	"strings"
)

const omniTurboPreset = "omni-turbo"

// WithLoRASampling resolves the preset on a request copy. The profile's
// regular sampling settings are kept on disk so switching back restores them.
func (p Profile) WithLoRASampling() Profile {
	if p.ModelID != OmniFP8RecipeID {
		return p
	}
	for _, lora := range p.LoRAs {
		if lora.Preset == omniTurboPreset && lora.Scale == 1 {
			p.Steps, p.CFGScale, p.Sampler = 8, 1, "euler"
			p.Scheduler = ""
			p.Sigmas = omniTurboSigmas(p.Width, p.Height, p.Steps)
			return p
		}
	}
	return p
}

func omniTurboSigmas(width, height, steps int) string {
	// Qwen Image 2.1 uses a 16x spatial VAE and passes the flattened target
	// latent grid size to Diffusers' FlowMatchEulerDiscreteScheduler. ChatR's
	// Turbo schedule uses dynamic exponential shifting with shift_terminal=None.
	tokens := float64((width / 16) * (height / 16))
	mu := 0.5 + (0.9-0.5)*(tokens-256)/(8192-256)
	shift := math.Exp(mu)
	sigmas := make([]string, steps+1)
	for i := range steps {
		t := float64(steps-i) / float64(steps)
		value := float32(shift / (shift + 1/t - 1))
		sigmas[i] = strconv.FormatFloat(float64(value), 'g', -1, 32)
	}
	sigmas[steps] = "0"
	return strings.Join(sigmas, ",")
}
