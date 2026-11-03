package server

import (
	"reflect"

	"github.com/ollama/ollama/api"
)

// Account conservatively for cloned Go containers, strings and boxed metadata.
// Large responses bypass retention without serializing another full copy just
// to measure it. The response returned by /api/show is never shortened.
func modelShowResponseBytes(response *api.ShowResponse, limit int) (int, bool) {
	if response == nil || limit <= 0 {
		return 0, false
	}
	remaining := uint64(limit)
	var visit func(reflect.Value, int) bool
	add := func(size uint64) bool {
		if size > remaining {
			return false
		}
		remaining -= size
		return true
	}
	containsPointers := func(kind reflect.Kind) bool {
		switch kind {
		case reflect.Array, reflect.Struct, reflect.Pointer, reflect.Interface, reflect.Map, reflect.Slice, reflect.String:
			return true
		default:
			return false
		}
	}
	visit = func(value reflect.Value, depth int) bool {
		if !value.IsValid() {
			return true
		}
		if depth > 64 || !add(uint64(value.Type().Size())) {
			return false
		}
		if value.Type() == reflect.TypeFor[api.ToolCallFunctionArguments]() && value.CanInterface() {
			if !add(128) {
				return false
			}
			arguments := value.Interface().(api.ToolCallFunctionArguments)
			for key, item := range arguments.All() {
				if !add(uint64(128+len(key))) || !visit(reflect.ValueOf(item), depth+1) {
					return false
				}
			}
			return true
		}
		switch value.Kind() {
		case reflect.String:
			return add(uint64(value.Len()))
		case reflect.Pointer, reflect.Interface:
			return value.IsNil() || visit(value.Elem(), depth+1)
		case reflect.Slice:
			if value.IsNil() {
				return true
			}
			item := uint64(value.Type().Elem().Size())
			count := uint64(value.Len())
			if item == 0 {
				return true
			}
			if item > 0 && count > remaining/item/2 {
				return false
			}
			if !add(count * item * 2) {
				return false
			}
			if !containsPointers(value.Type().Elem().Kind()) {
				return true
			}
			for i := 0; i < value.Len(); i++ {
				if !visit(value.Index(i), depth+1) {
					return false
				}
			}
		case reflect.Array:
			if !containsPointers(value.Type().Elem().Kind()) {
				return true
			}
			for i := 0; i < value.Len(); i++ {
				if !visit(value.Index(i), depth+1) {
					return false
				}
			}
		case reflect.Struct:
			for i := 0; i < value.NumField(); i++ {
				if !visit(value.Field(i), depth+1) {
					return false
				}
			}
		case reflect.Map:
			if value.IsNil() {
				return true
			}
			item := uint64(value.Type().Key().Size() + value.Type().Elem().Size())
			count := uint64(value.Len())
			if !add(64) || item > 0 && count > remaining/item/2 {
				return false
			}
			if !add(count * item * 2) {
				return false
			}
			iterator := value.MapRange()
			for iterator.Next() {
				if !visit(iterator.Key(), depth+1) || !visit(iterator.Value(), depth+1) {
					return false
				}
			}
		}
		return true
	}
	if !visit(reflect.ValueOf(response), 0) {
		return 0, false
	}
	return limit - int(remaining), true
}

func (c *modelShowCache) removeLocalLocked(key modelShowLocalKey) {
	if entry, ok := c.local[key]; ok {
		c.bytes -= entry.Bytes
		delete(c.local, key)
	}
}

func (c *modelShowCache) removeCloudLocked(key modelShowCloudKey) {
	if usage, ok := c.cloudUsage[key]; ok {
		c.bytes -= usage.Bytes
		delete(c.cloudUsage, key)
	}
	delete(c.cloud, key)
	delete(c.cloudNextReadRefreshAfter, key)
}

func (c *modelShowCache) reserveLocked(cost int) bool {
	if c.maxEntries <= 0 || cost <= 0 || cost > c.maxBytes {
		return false
	}
	for len(c.local)+len(c.cloud) >= c.maxEntries || c.bytes > c.maxBytes-cost {
		oldest := ^uint64(0)
		var localKey modelShowLocalKey
		var cloudKey modelShowCloudKey
		local := false
		for key, entry := range c.local {
			if entry.Used <= oldest {
				oldest, localKey, local = entry.Used, key, true
			}
		}
		for key, usage := range c.cloudUsage {
			if usage.Used <= oldest {
				oldest, cloudKey, local = usage.Used, key, false
			}
		}
		if local {
			c.removeLocalLocked(localKey)
		} else {
			c.removeCloudLocked(cloudKey)
		}
	}
	return true
}
