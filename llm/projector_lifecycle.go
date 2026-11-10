package llm

const projectorOnDemandEnvironment = "WILDLLAMA_MMPROJ_ON_DEMAND"

func setProjectorLifecycleEnvironment(environment map[string]string, projectors []string, supported bool) {
	// Do not inherit a lifecycle policy for an unproved or projector-free payload.
	environment[projectorOnDemandEnvironment] = ""
	if len(projectors) > 0 && supported {
		environment[projectorOnDemandEnvironment] = "1"
	}
}
