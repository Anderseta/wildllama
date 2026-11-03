package server

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"hash"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/gin-gonic/gin"
	"golang.org/x/image/webp"
)

const imageUploadLimit = 48 << 20

func (asset imageAssetRecord) relativePath() string {
	if asset.Input {
		return filepath.Join("inputs", asset.JobID, asset.Filename)
	}
	return filepath.Join(asset.JobID, asset.Filename)
}

func imageDecodeConfig(reader io.Reader, kind string) (image.Config, error) {
	switch kind {
	case "image/png":
		return png.DecodeConfig(reader)
	case "image/jpeg":
		return jpeg.DecodeConfig(reader)
	case "image/webp":
		return webp.DecodeConfig(reader)
	}
	return image.Config{}, errors.New("unsupported image media type")
}

func imageDecode(reader io.Reader, kind string) (image.Image, error) {
	switch kind {
	case "image/png":
		return png.Decode(reader)
	case "image/jpeg":
		return jpeg.Decode(reader)
	case "image/webp":
		return webp.Decode(reader)
	}
	return nil, errors.New("unsupported image media type")
}

func openImageAsset(asset imageAssetRecord) (*os.File, error) {
	validName := asset.Filename == "image.png" || !asset.Input && imageBatchOutputName(asset.Filename) || asset.Input && (asset.Filename == "image.jpg" || asset.Filename == "image.webp")
	if !imageSafeID(asset.JobID) || !validName {
		return nil, errors.New("invalid image asset identity")
	}
	root, err := os.OpenRoot(asset.OutputRoot)
	if err != nil {
		return nil, err
	}
	defer root.Close()
	file, err := root.Open(asset.relativePath())
	if err != nil {
		return nil, err
	}
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() > 128<<20 {
		file.Close()
		return nil, errors.New("invalid image asset file")
	}
	config, err := imageDecodeConfig(io.LimitReader(file, imageUploadLimit), asset.MIMEType)
	if err != nil || config.Width < 1 || config.Height < 1 || int64(config.Width)*int64(config.Height) > 64<<20 {
		file.Close()
		return nil, errors.New("invalid image asset contents")
	}
	_, err = file.Seek(0, io.SeekStart)
	if err != nil {
		file.Close()
		return nil, err
	}
	return file, nil
}

func imageBatchOutputName(filename string) bool {
	if !strings.HasPrefix(filename, "image-") || !strings.HasSuffix(filename, ".png") {
		return false
	}
	index := strings.TrimSuffix(strings.TrimPrefix(filename, "image-"), ".png")
	value, err := strconv.Atoi(index)
	return err == nil && value >= 2 && value <= 8 && strconv.Itoa(value) == index
}

func (s *Server) imageUploadHandler(c *gin.Context) {
	requestID, sessionID := c.Query("request_id"), c.Query("session_id")
	if requestID != "" && !imageSafeID(requestID) || len(sessionID) > 256 || !utf8.ValidString(sessionID) || strings.ContainsAny(sessionID, "\x00\r\n") || len(c.Request.URL.Query()["request_id"]) > 1 || len(c.Request.URL.Query()["session_id"]) > 1 {
		c.JSON(400, gin.H{"error": "invalid image upload request_id or session_id"})
		return
	}
	kind, _, err := mime.ParseMediaType(c.GetHeader("Content-Type"))
	extension := ""
	switch kind {
	case "image/png":
		extension = ".png"
	case "image/jpeg":
		extension = ".jpg"
	case "image/webp":
		extension = ".webp"
	}
	if err != nil || extension == "" {
		c.JSON(415, gin.H{"error": "upload a PNG, JPEG or WebP image with its image Content-Type"})
		return
	}
	if c.Request.ContentLength > imageUploadLimit {
		c.JSON(413, gin.H{"error": "reference images are limited to48 MiB"})
		return
	}
	v := s.imageService()
	// Full decoder validation is serialized. The64-megapixel bound is checked
	// before allocating pixels; concurrent uploads cannot multiply that cost.
	select {
	case v.uploadSlot <- struct{}{}:
		defer func() { <-v.uploadSlot }()
	case <-c.Request.Context().Done():
		return
	}
	v.mu.Lock()
	outputRoot := v.config.OutputDir
	existing, repeated := v.imageUploadByRequestIDLocked(requestID)
	v.mu.Unlock()
	if repeated {
		if existing.SessionID != sessionID {
			c.JSON(409, gin.H{"error": "image upload request_id belongs to a different session"})
			return
		}
		fingerprint, err := imageReadUpload(c, io.Discard, sessionID, kind)
		if err != nil {
			imageUploadReadError(c, err)
			return
		}
		if existing.UploadFingerprint != fingerprint {
			c.JSON(409, gin.H{"error": "image upload request_id belongs to different image bytes or media type"})
			return
		}
		if err := imageVerifyUploadedAsset(existing); err != nil {
			c.JSON(410, gin.H{"error": "the original uploaded image is unavailable or changed; submit a new request_id"})
			return
		}
		c.JSON(http.StatusOK, gin.H{"asset": existing.imageAsset, "replayed": true})
		return
	}
	id := "input-" + newImageID()
	filename := "image" + extension
	directory := filepath.Join(outputRoot, "inputs", id)
	if err = imageWriteAllowed(directory); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	if err = os.MkdirAll(directory, 0700); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	defer os.Remove(directory) // only removes this newly created directory if empty
	file, err := os.CreateTemp(directory, ".upload-*.tmp")
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	temporary := file.Name()
	defer os.Remove(temporary)
	fingerprint, err := imageReadUpload(c, file, sessionID, kind)
	if err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		imageUploadReadError(c, err)
		return
	}
	input, err := os.Open(temporary)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	config, err := imageDecodeConfig(io.LimitReader(input, imageUploadLimit), kind)
	if err != nil || config.Width <= 0 || config.Height <= 0 || int64(config.Width)*int64(config.Height) > 64<<20 {
		input.Close()
		c.JSON(400, gin.H{"error": "invalid reference image or dimensions above64 megapixels"})
		return
	}
	if _, err = input.Seek(0, io.SeekStart); err == nil {
		var decoded image.Image
		decoded, err = imageDecode(io.LimitReader(input, imageUploadLimit), kind)
		if err == nil && (decoded.Bounds().Dx() != config.Width || decoded.Bounds().Dy() != config.Height) {
			err = errors.New("image dimensions changed while decoding")
		}
	}
	input.Close()
	if err != nil {
		c.JSON(400, gin.H{"error": "reference image data is incomplete or invalid"})
		return
	}
	path := filepath.Join(directory, filename)
	if err = os.Rename(temporary, path); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	asset := imageAssetRecord{imageAsset: imageAsset{AssetID: id + "/" + filename, RequestID: requestID, SessionID: sessionID, JobID: id, Filename: filename, MIMEType: kind, Path: path, Width: config.Width, Height: config.Height}, OutputRoot: outputRoot, Input: true}
	if requestID != "" {
		asset.UploadFingerprint = fingerprint
	}
	v.mu.Lock()
	v.assets[asset.AssetID] = asset
	err = writeImageJSON(filepath.Join(v.root, "assets.json"), v.assets)
	if err != nil {
		delete(v.assets, asset.AssetID)
	}
	v.mu.Unlock()
	if err != nil {
		_ = os.Remove(path)
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusCreated, gin.H{"asset": asset.imageAsset})
}

func (v *imageService) imageUploadByRequestIDLocked(id string) (imageAssetRecord, bool) {
	if id != "" {
		for _, asset := range v.assets {
			if asset.Input && asset.RequestID == id {
				return asset, true
			}
		}
	}
	return imageAssetRecord{}, false
}

func imageUploadHash(sessionID, kind string) hash.Hash {
	digest := sha256.New()
	digest.Write([]byte("wildllama-image-upload-v1\x00"))
	for _, value := range []string{sessionID, kind} {
		var size [8]byte
		binary.LittleEndian.PutUint64(size[:], uint64(len(value)))
		digest.Write(size[:])
		digest.Write([]byte(value))
	}
	return digest
}

func imageReadUpload(c *gin.Context, target io.Writer, sessionID, kind string) (string, error) {
	digest := imageUploadHash(sessionID, kind)
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, imageUploadLimit)
	if _, err := io.Copy(io.MultiWriter(target, digest), c.Request.Body); err != nil {
		return "", err
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}

func imageVerifyUploadedAsset(asset imageAssetRecord) error {
	file, err := openImageAsset(asset)
	if err != nil {
		return err
	}
	defer file.Close()
	digest := imageUploadHash(asset.SessionID, asset.MIMEType)
	if _, err := io.Copy(digest, io.LimitReader(file, imageUploadLimit+1)); err != nil {
		return err
	}
	if hex.EncodeToString(digest.Sum(nil)) != asset.UploadFingerprint {
		return errors.New("uploaded image changed")
	}
	return nil
}

func imageUploadReadError(c *gin.Context, err error) {
	var max *http.MaxBytesError
	if errors.As(err, &max) {
		c.JSON(413, gin.H{"error": "reference images are limited to48 MiB"})
	} else {
		c.JSON(400, gin.H{"error": err.Error()})
	}
}
