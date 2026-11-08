package server

import (
	"errors"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func inferenceErrorPayload(err error) gin.H {
	if llm.IsOutOfMemory(err) {
		return gin.H{"error": "Falta de VRAM ou memória para executar o modelo ou expandir seu cache KV. O pedido foi interrompido sem nova tentativa automática. Libere memória ou reduza contexto/lote antes de tentar novamente.\n" + err.Error(),
			"status": http.StatusInsufficientStorage, "code": "out_of_memory", "retryable": false}
	}
	if strings.Contains(err.Error(), "model compute failed:") {
		return gin.H{"error": "Falha de cálculo no modelo. O pedido foi interrompido sem nova tentativa automática. Confira os logs e os recursos disponíveis.\n" + err.Error(),
			"status": http.StatusUnprocessableEntity, "code": "model_compute_failed", "retryable": false}
	}
	var status api.StatusError
	if errors.As(err, &status) {
		return gin.H{"error": status.ErrorMessage, "status": status.StatusCode}
	}
	return gin.H{"error": err.Error()}
}

func inferenceErrorBody(value gin.H, message string) gin.H {
	result := gin.H{"error": message}
	for _, key := range []string{"code", "retryable", "request_id"} {
		if field, ok := value[key]; ok {
			result[key] = field
		}
	}
	return result
}
