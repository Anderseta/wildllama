package llm

import "testing"

func TestRuntimeRequestIdentityNeverAttributesOverlappingSlots(t *testing.T) {
	w := NewStatusWriter(nil)
	first := w.beginRuntimeRequest("first")
	if got := w.ActivitySnapshot().RequestID; got != "first" {
		t.Fatalf("first identity %q", got)
	}
	second := w.beginRuntimeRequest("second")
	if got := w.ActivitySnapshot().RequestID; got != "" {
		t.Fatalf("overlap attributed to %q", got)
	}
	second()
	second() // idempotent cancellation/release
	if got := w.ActivitySnapshot().RequestID; got != "" {
		t.Fatalf("ambiguous history attributed to %q", got)
	}
	first()
	third := w.beginRuntimeRequest("third")
	if got := w.ActivitySnapshot().RequestID; got != "third" {
		t.Fatalf("new interval identity %q", got)
	}
	third()
	if got := w.ActivitySnapshot().RequestID; got != "" {
		t.Fatalf("idle retained identity %q", got)
	}
}
