package api

import "context"

type requestIDContextKey struct{}

// WithRequestID carries a validated correlation ID through native runner calls
// without adding it to model messages or prompt/cache keys.
func WithRequestID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, requestIDContextKey{}, id)
}

func RequestID(ctx context.Context) string {
	id, _ := ctx.Value(requestIDContextKey{}).(string)
	return id
}
