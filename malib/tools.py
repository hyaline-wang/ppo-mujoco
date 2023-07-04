
def ma_state_norm(state_norm, states, update=True):
    states_ = []
    for s in states:
        states_.append(state_norm(s, update=update))
    return states_

def ma_reward_norm(reward_norm, rewards):
    rewards_ = []
    for r in rewards:
        rewards_.append(reward_norm(r))
    return rewards_

def ma_reward_scaling(reward_scaling, rewards):
    rewards_ = []
    for r in rewards:
        rewards_.append(reward_scaling(r))
    return rewards_

def ma_choose_action(agent, states):
    a_ = []
    a_logprob_ = []
    for s in states:
        a, a_logprob = agent.choose_action(s)
        a_.append(a)
        a_logprob_.append(a_logprob)
    return a_, a_logprob_

def ma_evaluate(agent, states):
    a_ = []
    for s in states:
        a = agent.evaluate(s)
        a_.append(a)
    return a_

def ma_beta_action(actions, args):
    actions_ = []
    for a in actions:
        if args.policy_dist == "Beta":
            action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
        else:
            action = a
        actions_.append(action)
    return actions_

def ma_dw(terminateds, truncated):
    dw_ = []
    terminated = any(terminateds)
    for term in terminateds:
        if (terminated and term) and not truncated:
            dw_.append(True)
        else:
            dw_.append(False)
    return dw_