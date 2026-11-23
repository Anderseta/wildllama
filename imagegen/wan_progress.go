package imagegen

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

// The percentage estimates overall work. It advances on native events, never on a timer.
// The step counter comes from the sampler; segment counters subdivide an evaluation.
type wanProgress struct {
	stage                                                   string
	attempt, base, high, evaluations, evaluation, stepsDone int
	tile, tiles, chunk, chunks, segment, segments           int
	samplingSince                                           time.Time
	hostWeights                                             bool
}

func (w *wanProgress) restart(cfg float64) {
	attempt, base := w.attempt+1, w.high
	*w = wanProgress{attempt: attempt, base: base, high: base, evaluations: 1, tiles: 1}
	if cfg > 1 {
		w.evaluations = 2
	}
}

func (w *wanProgress) observe(line string, steps int) (Progress, bool) {
	if w.attempt == 0 {
		w.restart(1)
	}
	steps = max(1, steps)
	rawBar := strings.HasPrefix(line, "|") && !byteRatePattern.MatchString(line)
	var n, total int
	if rawBar || strings.Contains(line, "wildvideo ") {
		if match := stepPattern.FindStringSubmatch(line); match != nil {
			n, _ = strconv.Atoi(match[1])
			total, _ = strconv.Atoi(match[2])
			if total <= 0 || n > total {
				return Progress{}, false
			}
		}
	}
	switch {
	case strings.Contains(line, "auto-fit: --backend") && strings.Contains(line, "--params-backend") && strings.Contains(line, "=cpu"):
		w.hostWeights = true
		return Progress{Phase: "preparing", Message: "GPU selecionada para o cálculo, com pesos na RAM", VideoPercent: w.high, VideoAttempt: w.attempt, VideoHostWeights: true}, true
	case strings.Contains(line, "wan restored ") && strings.Contains(line, "mib of weights from ram"):
		if w.hostWeights || w.stage != "sample" {
			return Progress{}, false
		}
		w.hostWeights = true
	case strings.Contains(line, "sampling using ") && w.stage == "":
		w.stage = "prepare"
	case strings.Contains(line, "img2vid"):
		w.stage = "vision"
	case strings.Contains(line, "get_clip_vision_output completed"):
		w.stage = "reference"
	case strings.Contains(line, "encode_first_stage completed"), strings.Contains(line, "wildvideo stage text"):
		w.stage = "text"
	case strings.Contains(line, "wildvideo reference reused"):
		w.stage = "text"
		w.high = max(w.high, w.base+int(25*float64(100-w.base)/100))
		return Progress{Phase: "encoding", Message: "Referência recuperada da tentativa anterior", VideoPercent: w.high, VideoAttempt: w.attempt}, true
	case strings.Contains(line, "wildvideo stage negative"):
		w.stage = "negative"
	case strings.Contains(line, "generate_video ") && !strings.Contains(line, "completed"):
		w.stage, w.samplingSince = "sample", time.Now()
	case strings.Contains(line, "generating latent video completed"):
		w.stage, w.tile, w.tiles, w.chunk, w.chunks = "decode", 0, 1, 0, 0
	case strings.Contains(line, "decode_first_stage completed"), strings.Contains(line, "save result video to"):
		w.stage = "save"
	case total > 0 && strings.Contains(line, "wildvideo segment ") && w.stage == "sample":
		if n == 0 && w.segments > 0 && w.segment == w.segments {
			w.evaluation++
		}
		w.segment, w.segments = n, total
	case total > 0 && strings.Contains(line, "wildvideo tile ") && (w.stage == "reference" || w.stage == "decode"):
		w.tile, w.tiles, w.chunk, w.chunks = n, total, 0, 0
	case total > 0 && strings.Contains(line, "wildvideo chunk ") && (w.stage == "reference" || w.stage == "decode"):
		w.chunk, w.chunks = n, total
	case rawBar && total > 0 && w.stage == "sample" && total == steps:
		w.stepsDone, w.evaluation, w.segment, w.segments = n, 0, 0, 0
	case rawBar && total > 0 && (w.stage == "reference" || w.stage == "decode"):
		w.tile, w.tiles, w.chunk, w.chunks = n, total, 0, 0
	default:
		return Progress{}, false
	}

	var p Progress
	var local float64
	switch w.stage {
	case "prepare":
		p.Phase, p.Message = "preparing", "Preparando o gerador de vídeo"
	case "vision":
		p.Phase, p.Message, local = "encoding", "Lendo a imagem inicial com CLIP Vision", 2
	case "text", "negative":
		p.Phase, p.Message, local = "encoding", "Codificando o prompt com UMT5", 25
		if w.stage == "negative" {
			p.Message, local = "Codificando o prompt negativo com UMT5", 28
		}
	case "reference", "decode":
		f := float64(w.tile) + float64(w.chunk)/float64(max(1, w.chunks))
		p.Phase, p.Message, local = "encoding", "Codificando a referência com o VAE", 4+21*f/float64(w.tiles)
		if w.stage == "decode" {
			p.Phase, p.Message, local = "decoding", "Convertendo o vídeo em quadros com o VAE", 86+12*f/float64(w.tiles)
		}
		p.Message += fmt.Sprintf(" · área %d/%d", min(w.tile+1, w.tiles), w.tiles)
		if w.chunks > 1 {
			p.Message += fmt.Sprintf(" · bloco temporal %d/%d", min(w.chunk+1, w.chunks), w.chunks)
		}
	case "sample":
		eval := min(float64(w.evaluations), float64(w.evaluation)+float64(w.segment)/float64(max(1, w.segments)))
		f := min(1., (float64(w.stepsDone)+eval/float64(w.evaluations))/float64(steps))
		p.Phase, p.Current, p.Total = "sampling", w.stepsDone, steps
		p.Message = fmt.Sprintf("Difusão · passo %d/%d", min(steps, w.stepsDone+1), steps)
		if w.segments > 0 {
			p.Message += fmt.Sprintf(" · segmentos concluídos %d/%d", w.segment, w.segments)
		}
		local = 30 + 56*f
		if elapsed := time.Since(w.samplingSince).Seconds(); f >= 0.05 && elapsed >= 4 {
			p.VideoETASeconds = int(math.Ceil(elapsed / f * ((1 - f) + 12./56)))
		}
	case "save":
		p.Phase, p.Message, local = "saving", "Finalizando e gravando o arquivo WebM", 99
	}
	w.high = max(w.high, min(99, w.base+int(local*float64(100-w.base)/100)))
	p.VideoPercent, p.VideoAttempt = w.high, w.attempt
	p.VideoHostWeights = w.hostWeights
	return p, true
}
