package server

// cloneImageJobLocked detaches every public mutable reference. Network writes
// can then run without holding the service mutex or racing live progress.
func cloneImageJobLocked(job *imageJob) *imageJob {
	copy := *job
	copy.Results = append([]imageAsset(nil), job.Results...)
	copy.RestoreErrors = append([]string(nil), job.RestoreErrors...)
	if job.Result != nil {
		value := *job.Result
		copy.Result = &value
	}
	if job.Generation != nil {
		value := *job.Generation
		value.Profile = cloneImageProfile(value.Profile)
		value.ReferenceAssetIDs = append([]string(nil), value.ReferenceAssetIDs...)
		copy.Generation = &value
	}
	copy.cancel, copy.ctx = nil, nil
	return &copy
}
