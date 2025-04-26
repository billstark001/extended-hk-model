package recsys

import (
	"ehk-model/model"
)

type MixDump struct {
	RecSys1Dump any
	RecSys2Dump any
}

type Mix struct {
	model.HKModelRecommendationSystem
	Model       *model.HKModel
	RecSys1     model.HKModelRecommendationSystem
	RecSys2     model.HKModelRecommendationSystem
	RecSys1Rate float64
}

func _() model.HKModelRecommendationSystem {
	return &Mix{}
}

func (rs *Mix) PostInit(dumpData any) {
	if dumpData != nil {
		if data, ok := dumpData.(*MixDump); ok {
			rs.RecSys1.PostInit(&data.RecSys1Dump)
			rs.RecSys2.PostInit(&data.RecSys2Dump)
			return
		} else {
			panic("test")
		}
	}
	rs.RecSys1.PostInit(nil)
	rs.RecSys2.PostInit(nil)
}

func (rs *Mix) PreStep() {
	rs.RecSys1.PreStep()
	rs.RecSys2.PreStep()
}

func (rs *Mix) PreCommit() {
	rs.RecSys1.PreCommit()
	rs.RecSys2.PreCommit()
}

func (rs *Mix) PostStep(changed []*model.RewiringEventBody) {
	rs.RecSys1.PostStep(changed)
	rs.RecSys2.PostStep(changed)
}

func (rs *Mix) Recommend(
	agent *model.HKAgent,
	neighbors []*model.HKAgent,
	count int,
) []*model.TweetRecord {
	r1Count := int(float64(count)*rs.RecSys1Rate + 0.5)
	r2Count := count - r1Count
	ret := make([]*model.TweetRecord, 0)
	if r1Count > 0 {
		ret = append(ret, rs.RecSys1.Recommend(agent, neighbors, r1Count)...)
	}
	if r2Count > 0 {
		ret = append(ret, rs.RecSys2.Recommend(agent, neighbors, r2Count)...)
	}
	return ret
}

func (rs *Mix) Dump() any {
	return MixDump{
		RecSys1Dump: rs.RecSys1.Dump(),
		RecSys2Dump: rs.RecSys2.Dump(),
	}
}
