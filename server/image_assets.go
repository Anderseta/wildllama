package server

import (
	"errors"
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
	v.mu.Unlock()
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
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, imageUploadLimit)
	_, err = io.Copy(file, c.Request.Body)
	if err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		var max *http.MaxBytesError
		if errors.As(err, &max) {
			c.JSON(413, gin.H{"error": "reference images are limited to48 MiB"})
		} else {
			c.JSON(400, gin.H{"error": err.Error()})
		}
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
	asset := imageAssetRecord{imageAsset: imageAsset{AssetID: id + "/" + filename, JobID: id, Filename: filename, MIMEType: kind, Path: path, Width: config.Width, Height: config.Height}, OutputRoot: outputRoot, Input: true}
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
