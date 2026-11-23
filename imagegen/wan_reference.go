package imagegen

import "os"

// Only successive GPU attempts of the same job can access these exact VAE latents.
type wanReference struct {
	name  string
	file  *os.File
	close func()
}
