package imagegen

import "path/filepath"

const (
	mossOfficialRepo     = "OpenMOSS-Team/MOSS-Image-SFT"
	mossOfficialRevision = "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9"
	mossLoRARepo         = "axiomofmind/Moss-image-nsfw-uncensored-lora-GGUF"
	mossLoRARevision     = "a21e2465ff6a3ba099439fa9b8c4af708ed19761"
	mossFocusedLoRA      = "MOSS-Image-NSFW-Uncensored-LoRA-Alternative-1750-F16.gguf"
	mossOriginalLoRA     = "MOSS-Image-NSFW-Uncensored-LoRA-F16.gguf"
)

func mossRecipe() Recipe {
	p := DefaultProfile(MOSSRecipeID)
	p.LoRAs = []LoRA{{Path: filepath.Join("lora", mossFocusedLoRA), Scale: 0.8}}
	return Recipe{
		ID: MOSSRecipeID, Family: "moss-image",
		Name:         "MOSS-Image SFT · Q8 + Heretic + LoRAs NSFW",
		Capabilities: []string{"generate"}, DefaultProfile: p,
		Components: []Component{
			{Role: "diffusion_model", Repo: "axiomofmind/Moss-Image-SFT-GGUF", Revision: "d8a587c3a66bcab191e93bdd6f742d41b1104c68",
				Filename: "MOSS-Image-SFT-Q8_0.gguf", SHA256: "8349ebdd643f2c66da6b25789db7df3cc708a86c051c56207fde699156bfad8e", Size: 5705576416, Required: true},
			{Role: "text_encoder", Repo: "pottokao/MOSS-Image-Text-Encoder-Heretic-DHQ-GGUF", Revision: "341623999c894198e877a3573b53d683e0301efd",
				Filename: "moss-te-heretic-dhq-K-5.8G.gguf", SHA256: "d5d703a0f0bd37a557cd87e07eeefe433e018bd2bdeb285131a33f4eb984e2bc", Size: 5780046944, Required: true},
			{Role: "vae", Repo: mossOfficialRepo, Revision: mossOfficialRevision,
				Filename: "vae/diffusion_pytorch_model.safetensors", SHA256: "c3bdf94dc67ec69b30ea9fe1fa8340085679cae64d019f7c7e4a9921f1a7f17a", Size: 249521372, Required: true},
			{Role: "tokenizer", Repo: mossOfficialRepo, Revision: mossOfficialRevision,
				Filename: "tokenizer/tokenizer.json", SHA256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523", Size: 19989325, Required: true},
			{Role: "lora", Repo: mossLoRARepo, Revision: mossLoRARevision,
				Filename: mossFocusedLoRA, SHA256: "6366d060b9653c5b72c0694238a09b01787f3ca610e7f27af051a5aed8fff7d8", Size: 74365536, Required: true},
			{Role: "lora", Repo: mossLoRARepo, Revision: mossLoRARevision,
				Filename: mossOriginalLoRA, SHA256: "2c3e044a3331a94416ed91610ccd5135df0f79f0cb3d5db5983de0fb5dfa5603", Size: 74365536, Required: true},
		},
	}
}
