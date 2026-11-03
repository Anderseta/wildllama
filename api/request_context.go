package api

import "context"

type requestIDContextKey struct{}
type requestPriorityContextKey struct{}

// WithRequestID carries a validated correlation ID through native runner calls
// without adding it to model messages or prompt/cache keys.
func WithRequestID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, requestIDContextKey{}, id)
}

func RequestID(ctx context.Context) string {
	id, _ := ctx.Value(requestIDContextKey{}).(string)
	return id
}

// WithRequestPriority carries admission priority through adapters into model
// scheduling. It never changes the inference payload, sampling or cache key.
func WithRequestPriority(ctx context.Context, interactive bool) context.Context {
	return context.WithValue(ctx, requestPriorityContextKey{}, interactive)
}

func RequestPriority(ctx context.Context) (interactive, specified bool) {
	interactive, specified = ctx.Value(requestPriorityContextKey{}).(bool)
	return interactive, specified
}
