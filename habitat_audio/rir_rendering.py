import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import ActionSpaceConfiguration
from habitat.sims.habitat_simulator.actions import HabitatSimActions


# swapping PAUSE for STOP
HabitatSimActions.extend_action_space("PAUSE")
temp = HabitatSimActions.STOP
HabitatSimActions._known_actions["STOP"] = HabitatSimActions.PAUSE
HabitatSimActions._known_actions["PAUSE"] = temp
HabitatSimActions.extend_action_space("MOVE_FORWARD_COLLECT")
HabitatSimActions.extend_action_space("TURN_LEFT_COLLECT")
HabitatSimActions.extend_action_space("TURN_RIGHT_COLLECT")


@registry.register_action_space_configuration(name="rir-rendering")
class RIRRenderingActionSpaceConfiguration(ActionSpaceConfiguration):
    def get(self):        
        return {
            HabitatSimActions.PAUSE: habitat_sim.ActionSpec("pause"),
            HabitatSimActions.MOVE_FORWARD: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.TURN_LEFT: habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            HabitatSimActions.TURN_RIGHT: habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            HabitatSimActions.MOVE_FORWARD_COLLECT: habitat_sim.ActionSpec(
                "move_forward_collect",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.TURN_LEFT_COLLECT: habitat_sim.ActionSpec(
                "turn_left_collect",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            HabitatSimActions.TURN_RIGHT_COLLECT: habitat_sim.ActionSpec(
                "turn_right_collect",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
        }
    
@registry.register_action_space_configuration(name="move-grid-3x3")
class MoveGrid3x3SpaceConfiguration(ActionSpaceConfiguration):
    def get(self):
        return {
            HabitatSimActions.PAUSE: habitat_sim.ActionSpec("pause"),
            HabitatSimActions.TOP_L: habitat_sim.ActionSpec(
                "top_l",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.TOP_M: habitat_sim.ActionSpec(
                "top_m",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.TOP_R: habitat_sim.ActionSpec(
                "top_r",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.MID_L: habitat_sim.ActionSpec(
                "mid_l",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.MID_M: habitat_sim.ActionSpec(
                "mid_m",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.MID_R: habitat_sim.ActionSpec(
                "mid_r",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.BOT_L: habitat_sim.ActionSpec(
                "bot_l",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.BOT_M: habitat_sim.ActionSpec(
                "bot_m",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.BOT_R: habitat_sim.ActionSpec(
                "bot_r",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            )
        } 