package api

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"testing"
)

type emptyStreamBody struct {
	readErr error
	closed  bool
}

func (b *emptyStreamBody) Read([]byte) (int, error) {
	if b.readErr != nil {
		return 0, b.readErr
	}
	return 0, io.EOF
}

func (b *emptyStreamBody) Close() error {
	b.closed = true
	return nil
}

func TestClientStreamEmptyResponseCompatibility(t *testing.T) {
	for _, tc := range []struct {
		status  int
		readErr error
	}{
		{status: http.StatusOK},
		{status: http.StatusNoContent},
		{status: http.StatusBadRequest},
		{status: http.StatusUnauthorized},
		{status: http.StatusNotFound},
		{status: http.StatusTooManyRequests},
		{status: http.StatusInternalServerError},
		{status: http.StatusInternalServerError, readErr: context.Canceled},
		{status: http.StatusInternalServerError, readErr: io.ErrUnexpectedEOF},
	} {
		t.Run(fmt.Sprintf("%d/%v", tc.status, tc.readErr), func(t *testing.T) {
			body := &emptyStreamBody{readErr: tc.readErr}
			statusText := fmt.Sprintf("%d %s", tc.status, http.StatusText(tc.status))
			client := NewClient(&url.URL{Scheme: "http", Host: "localhost"}, &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: tc.status, Status: statusText, Body: body, Header: make(http.Header), Request: r}, nil
			})})
			err := client.stream(t.Context(), http.MethodPost, "/api/chat", nil, func([]byte) error {
				t.Error("empty response reached callback")
				return nil
			})
			if !body.closed {
				t.Fatal("empty response body was not closed")
			}
			if tc.readErr != nil {
				if !errors.Is(err, tc.readErr) {
					t.Fatalf("read error changed: %v", err)
				}
				return
			}
			if tc.status < http.StatusBadRequest {
				if err != nil {
					t.Fatalf("empty success changed: %v", err)
				}
				return
			}
			if tc.status == http.StatusUnauthorized {
				var auth AuthorizationError
				if !errors.As(err, &auth) || auth.StatusCode != tc.status || auth.Status != statusText {
					t.Fatalf("authorization metadata lost: %T %v", err, err)
				}
			} else {
				var failure StatusError
				if !errors.As(err, &failure) || failure.StatusCode != tc.status || failure.Status != statusText {
					t.Fatalf("HTTP metadata lost: %T %v", err, err)
				}
			}
		})
	}
}
